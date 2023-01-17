import numpy as np 

import pandas as pd 

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

import matplotlib as mpl

import matplotlib.pylab as pylab

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import random

import time

import warnings

warnings.filterwarnings('ignore')

print('-'*25)

from subprocess import check_output
df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
df.head()
df.shape
df.describe()
# checking missing data

total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(30)
#duplicate check

df.drop_duplicates()
df.dropna(axis=0, how='any')
fig, saxis = plt.subplots(2, 3,figsize=(20,20))



sns.countplot(x ='TARGET',data=df, ax = saxis [0,0])

sns.countplot(x='NAME_CONTRACT_TYPE',data=df, ax = saxis [0,1])

sns.countplot(x='CODE_GENDER',data=df, ax = saxis [0,2])

              

sns.countplot(x ='FLAG_OWN_CAR',data=df, ax = saxis [1,0])

sns.countplot(x='FLAG_OWN_REALTY',data=df, ax = saxis [1,1])

sns.countplot(x='CNT_FAM_MEMBERS',data=df, ax = saxis [1,2])
plt.figure(figsize=(7,5))

sns.distplot(df.loc[df['AMT_INCOME_TOTAL'] < 0.25e7, 'AMT_INCOME_TOTAL'].dropna())

plt.show()
plt.figure(figsize=(12,5))

plt.title("AMT_CREDIT")

ax = sns.distplot(df["AMT_CREDIT"])
plt.figure(figsize=(12,5))

plt.title("AMT_ANNUITY")

ax = sns.distplot(df["AMT_ANNUITY"])
fig, ax = plt.subplots(figsize=(11,7))

sns.countplot(x='NAME_INCOME_TYPE',data=df)

plt.xlabel("NAME_INCOME_TYPE")

plt.xticks(rotation=70)

plt.title('NAME_INCOME_TYPE', fontsize=20)
fig, ax = plt.subplots(figsize=(13,5))

sns.countplot(x='NAME_EDUCATION_TYPE',data=df)

plt.xlabel("NAME_EDUCATION_TYPE")

plt.xticks(rotation=70)
fig, ax = plt.subplots(figsize=(11,7))

sns.countplot(x='NAME_FAMILY_STATUS',data=df)

plt.xlabel("NAME_FAMILY_STATUS")

plt.xticks(rotation=70)

plt.title('NAME_FAMILY_STATUS', fontsize=20)
fig, saxis = plt.subplots(2, 4,figsize=(20,20))



sns.countplot(x='TARGET',hue='NAME_INCOME_TYPE',data=df, ax = saxis [0,0])

sns.countplot(x='TARGET',hue='CODE_GENDER',data=df, ax = saxis [0,1])

sns.countplot(x='TARGET',hue='FLAG_OWN_CAR',data=df, ax = saxis [0,2])

sns.countplot(x='TARGET',hue='FLAG_OWN_REALTY',data=df, ax = saxis [0,3])



sns.countplot(x='TARGET',hue='NAME_CONTRACT_TYPE',data=df, ax = saxis [1,0])

sns.countplot(x='TARGET',hue='NAME_FAMILY_STATUS',data=df, ax = saxis [1,1])

sns.countplot(x='TARGET',hue='OCCUPATION_TYPE',data=df, ax = saxis [1,2])

sns.countplot(x='TARGET',hue='NAME_EDUCATION_TYPE',data=df, ax = saxis [1,3])
columns_to_keep = missing_data[missing_data['Percent'] < 10]

columns_to_keep.index.tolist()
df = df[columns_to_keep.index.tolist()]
df.shape
df.isnull().sum()
df = df.dropna()

df.shape
df = df.set_index('SK_ID_CURR')
# Change all negative values

df['DAYS_LAST_PHONE_CHANGE'] = abs(df['DAYS_LAST_PHONE_CHANGE'])

df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])

df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])

df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])



# Change the days from birth to age

df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH']) // 365

df.head()
df = df.replace(365243, np.nan)

df = df.replace('XNA', np.nan)
df.DAYS_EMPLOYED.isnull().sum()
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].interpolate()
df.DAYS_EMPLOYED.isnull().sum()
#Replace categorical values to binary

df = df.replace({'FLAG_OWN_CAR' : { 'N' : 0, 'Y' : 1 }, 'CODE_GENDER' : { 'M' : 0, 'F' : 1 }, 'FLAG_OWN_REALTY' : { 'N' : 0, 'Y' : 1 }})
df.FLAG_OWN_CAR.unique()
df.NAME_EDUCATION_TYPE.unique()
df.WEEKDAY_APPR_PROCESS_START.unique()
#Change categorical ordinal variables to numeric

df = df.replace({'NAME_EDUCATION_TYPE' : { 'Lower secondary' : 0, 'Secondary / secondary special' : 1, 'Incomplete higher': 2, 'Higher education': 3, 'Academic degree': 4 }, 'WEEKDAY_APPR_PROCESS_START' : { 'MONDAY' : 0, 'TUESDAY' : 1 , 'WEDNESDAY': 2, 'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6}})
df = pd.get_dummies(data=df, columns=['NAME_TYPE_SUITE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', \

                             'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE'])
df_to_scale = df[['AMT_GOODS_PRICE','AMT_ANNUITY', 'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT','AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH','DAYS_REGISTRATION', 'DAYS_EMPLOYED','DAYS_BIRTH', 'HOUR_APPR_PROCESS_START']]
ct = ColumnTransformer([



        ('somename', StandardScaler(), df_to_scale.columns.tolist())



    ], remainder='passthrough')



scaled_data = pd.DataFrame(ct.fit_transform(df_to_scale), columns=df_to_scale.columns.tolist())

scaled_data.head()
scaled_data.index = df.index.copy()

df = df.drop(columns=df_to_scale.columns.tolist())

print(scaled_data.shape)

print(df.shape)
df1 = pd.concat([df, scaled_data], axis=1)

df1.head()
df1.shape
df1.isnull().sum()
df1 = df1.dropna()
X = df1.loc[:, df1.columns != 'TARGET']

Y = df1.loc[:, df1.columns == 'TARGET']



X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.7)
lr = LogisticRegression()



lr.fit(X_train, Y_train)



Y_pred_class = lr.predict(X_test)



print(classification_report(Y_test, Y_pred_class))



print(roc_auc_score(Y_test, Y_pred_class))
dt = DecisionTreeClassifier(random_state=42)



dt.fit(X_train, Y_train)



Y_pred_class = dt.predict(X_test)



print(classification_report(Y_test, Y_pred_class))



print(roc_auc_score(Y_test, Y_pred_class))
rf = RandomForestClassifier(max_depth=5, random_state=42)



rf.fit(X_train, Y_train)



Y_pred_class = rf.predict(X_test)



print(classification_report(Y_test, Y_pred_class))



print(roc_auc_score(Y_test, Y_pred_class))
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)



gbm.fit(X_train, Y_train)



Y_pred_class = gbm.predict(X_test)



print(classification_report(Y_test, Y_pred_class))



print(roc_auc_score(Y_test, Y_pred_class))
submission = pd.DataFrame(Y_pred_class, columns=['TARGET'], index = X_test.index)



submission.to_csv('submission.csv')
submission