import numpy as np 

import pandas as pd 



# for standardization

from sklearn.preprocessing import StandardScaler as ss



# for splitting into train and test datasets

from sklearn.model_selection import train_test_split 



# for modelling

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier



# for balancing dataset by oversampling

from imblearn.over_sampling import SMOTE, ADASYN



# for performance metrics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, classification_report



# for data visualization

import matplotlib.pyplot as plt



#Miscellaneous

import time

import random

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# set number of rows to be displayed

pd.options.display.max_columns = 300



# reading the dataset

data = pd.read_csv("../input/creditcard.csv")
#Shape

data.shape

#Columns

data.columns.values
# check if there are null values in the dataset

data.isnull().sum().sum()
#datatypes

data.dtypes.value_counts()
#Checking unbalance data

data.Class.value_counts()
data.head()

data.tail()
import seaborn as sns

sns.countplot(data['Class'])
#Splitting Data as features and target
X = data.iloc[:,0:30]

y = data.iloc[:,30]
X_train, X_test, y_train, y_test =   train_test_split(X, y, test_size = 0.3, stratify = y)
X_train.shape
y_train.value_counts()
sm = SMOTE(random_state=42)

X_bal, y_bal = sm.fit_sample(X_train, y_train)



columns = X_train.columns

X_bal = pd.DataFrame(data = X_bal, columns = columns)

print(X_train.shape)

print(X_bal.shape)

print(np.unique(y_bal, return_counts=True))
# Intantiating RandomForest and XGBoost Models to train balanced data

rf_sm = RandomForestClassifier(n_estimators=80)

xg_sm = XGBClassifier(learning_rate=0.8,

                   reg_alpha= 0.7,

                   reg_lambda= 0.5

                   )



# training the models

rf_sm1 = rf_sm.fit(X_bal,y_bal)

xg_sm1 = xg_sm.fit(X_bal,y_bal)
# Predictions on the test data

#For Random Forest

y_pred_rf1 = rf_sm1.predict(X_test)

#For XQBoost

y_pred_xg1= xg_sm1.predict(X_test)
#Probability

y_pred_rf_prob1 = rf_sm1.predict_proba(X_test)

y_pred_xg_prob1 = xg_sm1.predict_proba(X_test)
#fpr and tpr

#For Random Forest

fpr_rf, tpr_rf, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob1[: , 1],

                                 pos_label= 1

                                 )



#For XQBoost

fpr_xg, tpr_xg, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob1[: , 1],

                                 pos_label= 1

                                 )
#Precision, Recall And F1 Score for RF and XGBoost

p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf1)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg1)

print("Accuracy of Randaom Forest : ",accuracy_score(y_test,y_pred_rf1))

print("Accuracy of XGBoost        : ",accuracy_score(y_test,y_pred_xg1))

print("Confusion Matrix for RF    :\n ",confusion_matrix(y_test,y_pred_rf1))

print("Confusion Matrix XGBoost   :\n ",confusion_matrix(y_test,y_pred_xg1))

print("AUC for Randaom Forest     : ",auc(fpr_rf,tpr_rf))

print("AUC for XGBoost            : ",auc(fpr_xg,tpr_xg))

print("Random Forest Precision, Recall and F1 Score are :")

print(p_rf)

print(r_rf)

print(f_rf)



print("XGBoost Precision, Recall and F1 Score are :")

print(p_xg)

print(r_xg)

print(f_xg)
ad = ADASYN()

X_ad, y_ad = ad.fit_sample(X_train, y_train)

 

X_ad = pd.DataFrame(data = X_ad, columns = X_train.columns)

#Dataset is balanced now

print(X_train.shape)

print(X_ad.shape)

print(np.unique(y_ad, return_counts=True))
# Intantiating RandomForest and XGBoost Models to train balanced data

rf_ad = RandomForestClassifier(n_estimators=80)

xg_ad = XGBClassifier(learning_rate=0.9,

                   reg_alpha= 0.8,

                   reg_lambda= 1

                   )



# training the models

rf_ad1 = rf_ad.fit(X_ad,y_ad)

xg_ad1 = xg_ad.fit(X_ad,y_ad)
# Predictions on the test data

#For Random Forest

y_pred_rf2 = rf_ad1.predict(X_test)

#For XQBoost

y_pred_xg2= xg_ad1.predict(X_test)
#Probability

y_pred_rf_prob2 = rf_ad1.predict_proba(X_test)

y_pred_xg_prob2 = xg_ad1.predict_proba(X_test)
#fpr and tpr

#For Random Forest

fpr_rf1, tpr_rf1, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob2[: , 1],

                                 pos_label= 1

                                 )



#For XQBoost

fpr_xg1, tpr_xg1, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob2[: , 1],

                                 pos_label= 1

                                 )
#Precision, Recall And F1 Score for RF and XGBoost

p_rf2,r_rf2,f_rf2,_ = precision_recall_fscore_support(y_test,y_pred_rf2)

p_xg2,r_xg2,f_xg2,_ = precision_recall_fscore_support(y_test,y_pred_xg2)
print("Accuracy of Randaom Forest : ",accuracy_score(y_test,y_pred_rf2))

print("Accuracy of XGBoost        : ",accuracy_score(y_test,y_pred_xg2))

print("Confusion Matrix for RF    :\n ",confusion_matrix(y_test,y_pred_rf2))

print("Confusion Matrix XGBoost   :\n ",confusion_matrix(y_test,y_pred_xg2))

print("AUC for Randaom Forest     : ",auc(fpr_rf1,tpr_rf1))

print("AUC for XGBoost            : ",auc(fpr_xg1,tpr_xg1))

print("Random Forest Precision, Recall and F1 Score are :")

print(p_rf2)

print(r_rf2)

print(f_rf2)



print("XGBoost Precision, Recall and F1 Score are :")

print(p_xg2)

print(r_xg2)

print(f_xg2)