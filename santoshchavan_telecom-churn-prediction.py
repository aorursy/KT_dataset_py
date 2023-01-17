import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # For creating plots

import matplotlib.ticker as mtick # For specifying the axes tick format 

import matplotlib.pyplot as plt



sns.set(style = 'white')



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
telecom_cust = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
telecom_cust.head()
#Feature Extraction

X = telecom_cust.drop(['customerID','Churn'], axis = 1)
#Label Extraction



y = telecom_cust['Churn'].copy()
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
#Replacing Spaces with Nan

X_train['TotalCharges'].replace(' ', np.NaN,inplace = True)

X_test['TotalCharges'].replace(' ',np.NaN, inplace = True)
#Converting the type of column

X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)

X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)
#Filling missing values

X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(),inplace = True)

X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(),inplace = True)
print(X_train.isnull().sum())
cat_cols = X_train.select_dtypes(include = 'O').columns.tolist()

print(cat_cols)
from sklearn.preprocessing import LabelEncoder

for x in cat_cols:

    le = LabelEncoder()

    X_train[x] = le.fit_transform(X_train[x])
for x in cat_cols:

    le = LabelEncoder()

    X_test[x] = le.fit_transform(X_test[x])
#Encoding train data target    

y_train = y_train.replace({'No':0, 'Yes':1})



y_test = y_test.replace({'No':0, 'Yes':1})
X_train.head()
X_test.head()
y_train.head()
y_test.head()
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(random_state = 0)

ada_model.fit(X_train,y_train)

y_pred = ada_model.predict(X_test)

print(y_pred)



e = ada_model.score(X_test,y_test)

print(e)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



ada_score = accuracy_score(y_test,y_pred)



print(ada_score)
ada_cm = confusion_matrix(y_test,y_pred)

ada_cm
ada_cr = classification_report(y_test, y_pred)

print(ada_cr)
from xgboost import XGBClassifier



#parameter list 

parameters= {'learning_rate':[0.1,0.15,0.2,0.25,0.3],'max_depth':range(1,3)}



xgb_model = XGBClassifier(random_state =0)

xgb_model.fit(X_train,y_train)

score = xgb_model.score(X_test,y_test)

print(score)
y_pred = xgb_model.predict(X_test)



xgb_score = accuracy_score(y_test,y_pred)



print(xgb_score)
xgb_cm = confusion_matrix(y_test,y_pred)

xgb_cm
xgb_cr = classification_report(y_test,y_pred)

print(xgb_cr)
from sklearn.model_selection import GridSearchCV
X_test.shape
y_test.shape
clf_model = GridSearchCV(xgb_model,parameters)

clf_model.fit(X_train,y_train)

y_pred = clf_model.predict(X_test)



clf_score = accuracy_score(y_test, y_pred)

print(clf_score)



clf_cm = confusion_matrix(y_test, y_pred)

print(clf_cm)



clf_cr = classification_report(y_test, y_pred)

print(clf_cr)

print(y_pred)