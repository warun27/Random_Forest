# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 01:05:06 2020

@author: shara
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
company = pd.read_csv("G:\DS Assignments\Random_forest\Company_Data.csv")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
company.info()
Sales_catg = []

a = "<5"
b = ">=5 and <10"
c = ">10"

for i in company.Sales:
    if i < 5:
        Sales_catg.append(a)
    elif i >= 5 and i < 10:
        Sales_catg.append(b)
    else:
        Sales_catg.append(c)
    
company["Sales_catg"] = Sales_catg
company["Sales_catg"].value_counts()
categorical_column = ["ShelveLoc", "Urban","US" ]
company_dummy = pd.get_dummies(company, columns = categorical_column)
company_dummy = company_dummy.drop("Sales", axis = 1)
x = company_dummy.drop("Sales_catg", axis = 1)
y = company_dummy["Sales_catg"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20,random_state = 7)

rf = RandomForestClassifier(n_jobs = 2, oob_score=True, n_estimators = 30, criterion = "entropy")
# rf.fit(x_train,y_train)
# y_pred = rf.predict(x_train)
from sklearn.metrics import confusion_matrix
# confusion_matrix(y_train, y_pred,)
from sklearn.metrics import classification_report
# cls_report= pd.DataFrame(classification_report(y_train, y_pred, output_dict=(True)))
# y_test_pred = rf.predict(x_test)
# confusion_matrix(y_test, y_test_pred)


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 0)
x_smote, y_smote = smote.fit_resample(x,y)
y_smote.value_counts()
x_train_smote,x_test_smote,y_train_smote,y_test_smote = train_test_split(x_smote,y_smote, test_size = 0.20,random_state = 7)
# rf.fit(x_train_smote,y_train_smote)
# y_pred_smote = rf.predict(x_train_smote)
# y_test_pred_smote = rf.predict(x_test_smote)
# confusion_matrix(y_test_smote, y_test_pred_smote)
# confusion_matrix(y_train_smote, y_pred_smote)
from sklearn import preprocessing
x_train_scaled = preprocessing.scale(x_train_smote)
x_test_scaled = preprocessing.scale(x_test_smote)

rf.fit(x_train_scaled,y_train_smote)
y_pred_scaled = rf.predict(x_train_scaled)
y_test_pred_scaled = rf.predict(x_test_scaled)
confusion_matrix(y_test_smote, y_test_pred_scaled)
confusion_matrix(y_train_smote, y_pred_smote)
np.mean(y_test_smote == y_test_pred_scaled)
cls_report = pd.DataFrame(classification_report(y_test_pred_scaled, y_test_smote, output_dict=(True)))
