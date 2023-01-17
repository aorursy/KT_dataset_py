import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.shape
df.head()
df.dtypes
pd.to_numeric(df['TotalCharges'])
df.iloc[488]
df = df.replace('^\s*$',np.nan, regex = True)
df.isnull().sum()
df.shape
df.dropna(axis = 0 ,inplace = True)

df.shape #11 missing observations are removed
df.isnull().sum()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['Churn'].value_counts()
df.head()
col = ['Partner','Dependents','PhoneService','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
col_2 = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
df['Partner'].value_counts()
col_3 = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df['StreamingMovies'].value_counts()
df['gender'].replace(('Male','Female'),(1,0),inplace = True)
df_2 = df[col_2]
df_2.head()
for i,j in enumerate(df_2.columns):

    df[j] = df[j].replace(('Yes','No'),(1,0))
df.head()
df_3 = df[col_3]
for i,j in enumerate(df_3.columns):

    df[j] = df[j].replace(('No internet service','No','Yes'),(0,1,2))
df['MultipleLines'].value_counts()
df['MultipleLines'].replace(('No phone service','No','Yes'),(0,1,2),inplace = True)
df['InternetService'].value_counts()
df['InternetService'].replace(('No','DSL','Fiber optic'),(0,1,2),inplace = True)
df['Contract'].value_counts()
df['Contract'].replace(('Month-to-month','One year','Two year'),(1,2,3),inplace = True)
df['PaymentMethod'].value_counts()
del df['customerID']
df.isnull().sum()
df.head()
df.dtypes
df_final = pd.get_dummies(df)
df_final.shape
df_final.columns
y = df_final['Churn']

X = df_final.drop('Churn', axis = 1)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score,f1_score,confusion_matrix
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size =0.2, shuffle = True)
print(X_train.shape)

print(y_train.shape)
lr = LogisticRegression().fit(X_train,y_train)

lr_pred = lr.predict(X_test)
lr_precision = precision_score(y_test,lr_pred)

lr_recall = recall_score(y_test,lr_pred)

lr_f1 = f1_score(y_test,lr_pred)
print('LR Precision',lr_precision)

print('LR recall', lr_recall)

print('LR F1 score',lr_f1)
rf = RandomForestClassifier(random_state =42).fit(X_train,y_train)

rf_pred = rf.predict(X_test)
rf_precision = precision_score(y_test,rf_pred)

rf_recall = recall_score(y_test,rf_pred)

rf_f1 = f1_score(y_test,rf_pred)
print('RF Precision',rf_precision)

print('RF recall', rf_recall)

print('RF F1 score',rf_f1)
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state =42)
rf_params = {

    'min_samples_split':[2,3,4],

    'min_samples_leaf':[1,2],

    'n_estimators' : [100,150,200]

}
GridSearchCV(rf, param_grid = rf_params,verbose = True).fit(X,y).best_params_
rf = RandomForestClassifier(min_samples_leaf= 2, min_samples_split=2,n_estimators=150).fit(X_train,y_train)
rf_pred = rf.predict(X_test)
rf_precision = precision_score(y_test,rf_pred)

rf_recall = recall_score(y_test,rf_pred)

rf_f1 = f1_score(y_test,rf_pred)
print('RF Precision',rf_precision)

print('RF recall', rf_recall)

print('RF F1 score',rf_f1)
val = rf.feature_importances_
val
imp_var = []

imp_var_val = []

for i,j in zip(X.columns,rf.feature_importances_):

    if j > 0.02:

        imp_var.append(j)

        imp_var_val.append(i)    
col_name = X.columns

plt.barh(col_name,val)

plt.xlabel('RF feature importance')

plt.show()
y = df['Churn']

X_new = df_final[['tenure','TotalCharges','MonthlyCharges', 'Contract']]
X_train,X_test,y_train,y_test = train_test_split(X_new,y, test_size =0.2, shuffle = True)
lr = LogisticRegression().fit(X_train,y_train)

lr_pred = lr.predict(X_test)
lr_precision = precision_score(y_test,lr_pred)

lr_recall = recall_score(y_test,lr_pred)

lr_f1 = f1_score(y_test,lr_pred)
print('LR Precision',lr_precision)

print('LR recall', lr_recall)

print('LR F1 score',lr_f1)