
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.isna().sum()
totalmiss=pd.DataFrame(data=[df.columns,df.isna().sum()/df.shape[0]])
totalmiss

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
oe=OrdinalEncoder()
si=SimpleImputer(strategy='most_frequent')
df=pd.DataFrame(si.fit_transform(df),columns=list(df))
df=pd.DataFrame(oe.fit_transform(df),columns=list(df))
df.info()
df.corr()['BAD'].sort_values(ascending=False)
x_train=df.drop(columns=['BAD'],axis=1)
y_train=df['BAD']
q=pd.DataFrame(data=[y_train.value_counts(),y_train.value_counts()/y_train.shape[0]])
print(q)
from imblearn.over_sampling import SMOTE
smt=SMOTE()
x_train,y_train=smt.fit_resample(x_train,y_train)
q=pd.DataFrame(data=[y_train.value_counts(),y_train.value_counts()/y_train.shape[0]])
print(q)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

print('Accuracy Score :',accuracy_score(y_test,y_pred))
print('F1_Score :',f1_score(y_test,y_pred))
print('ROC_AUC_Score',roc_auc_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

import xgboost
xgb=xgboost.XGBClassifier(max_depth=10,n_estimators=100)
xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)


print('Accuracy Score :',accuracy_score(y_test,y_pred))
print('F1_Score :',f1_score(y_test,y_pred))
print('ROC_AUC_Score',roc_auc_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
import lightgbm
lgbm=lightgbm.LGBMClassifier()
lgbm.fit(x_train,y_train)
y_pred=lgbm.predict(x_test)

print('Accuracy Score :',accuracy_score(y_test,y_pred))
print('F1_Score :',f1_score(y_test,y_pred))
print('ROC_AUC_Score',roc_auc_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))