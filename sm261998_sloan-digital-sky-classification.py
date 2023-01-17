import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier



#from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
df = pd.read_csv('../input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv')

df.head(10)
df.shape
df.keys()
df.isnull().sum()
df.describe()
print(df['class'].unique())

sns.countplot(df['class'])
df['run'].unique()
df['objid'].unique()
for i in df.keys():

    print("Colname:=",i)

    print(df[i].unique())
df.corr()
dictionary={'STAR':1,'GALAXY':2,'QSO':3}

df.replace({'class':dictionary},inplace=True)



# you can use LabelEncoder here also
df.head()
y = df['class']
df = df.drop(['class','rerun'], axis=1)
df.drop('objid',axis=1)
df.head(5)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

sdss = scaler.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(sdss, y, test_size=0.2, random_state=2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Logistic Regression

lr = LogisticRegression(C=2, max_iter=1500)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print("Accuracy of Logistic Regression= ", accuracy_score(y_test,y_pred))
# KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy of KNeighborsClassifier = ", accuracy_score(y_test,y_pred))
# RandomForestClassifier

rf = RandomForestClassifier(max_depth=18, n_estimators=120)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy of RandomForestClassifier = ", accuracy_score(y_test,y_pred))
# XGBClassifier

xgb= XGBClassifier()

xgb.fit(X_train,y_train)

y_pred =xgb.predict(X_test)

print("Accuracy of XGBClassifier = ", accuracy_score(y_test,y_pred))
import lightgbm as lgb



lgb=lgb.LGBMClassifier()

lgb.fit(X_train,y_train)

y_pred =lgb.predict(X_test)

print("Accuracy of lightgbm = ", accuracy_score(y_test,y_pred))
# GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train,y_train)

y_pred =gb.predict(X_test)

print("Accuracy of GradientBoostingClassifier = ", accuracy_score(y_test,y_pred))