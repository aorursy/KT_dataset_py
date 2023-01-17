# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

from matplotlib import pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/weatherAUS.csv')
data.head()
len(data)
data.shape
data.RainToday.value_counts()
data.RainTomorrow.value_counts()
# Filling missing values
data.isnull().sum()
(data.isnull().sum() / len(data)).sort_values(ascending=False)
data.columns
# Removing columns with too much null values

data = data.drop(['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'], axis=1)
# For the plots

data_no_nan = data.dropna()
# Pressure9am

data.Pressure9am.describe()
plt.figure(figsize=(10,6))

sns.distplot(data_no_nan.Pressure9am)
# Fill with mean, nearly same as median

data.Pressure9am = data.Pressure9am.fillna(data.Pressure9am.mean())
# Pressure3pm

data.Pressure3pm.describe()
plt.figure(figsize=(10,6))

sns.distplot(data_no_nan.Pressure3pm)
# Fill with mean, nearly same as median

data.Pressure3pm = data.Pressure3pm.fillna(data.Pressure3pm.mean())
# WindDir9am

data.WindDir9am.value_counts()
plt.figure(figsize=(10,6))

sns.countplot(data_no_nan.WindDir9am)
# Fill with max value

data.WindDir9am = data.WindDir9am.fillna(data.WindDir9am.value_counts().reset_index().iloc[0]['index'])
# WindGustDir

data.WindGustDir.value_counts()
plt.figure(figsize=(10,6))

sns.countplot(data_no_nan.WindGustDir)
# Fill with max value

data.WindGustDir = data.WindGustDir.fillna(data.WindGustDir.value_counts().reset_index().iloc[0]['index'])
# WindGustSpeed

data.WindGustSpeed.describe()
plt.figure(figsize=(10,6))

sns.distplot(data_no_nan.WindGustSpeed)
# Fill with mean, nearly same as median

data.WindGustSpeed = data.WindGustSpeed.fillna(data.WindGustSpeed.mean())
# WindDir3pm

data.WindDir3pm.value_counts()
plt.figure(figsize=(10,6))

sns.countplot(data_no_nan.WindDir3pm)
# Fill with max value

data.WindDir3pm = data.WindDir3pm.fillna(data.WindDir3pm.value_counts().reset_index().iloc[0]['index'])
# Humidity3pm

data.Humidity3pm.describe()
plt.figure(figsize=(10,6))

sns.distplot(data_no_nan.Humidity3pm)
# Fill with mean, nearly same as median

data.Humidity3pm = data.Humidity3pm.fillna(data.Humidity3pm.mean())
# Temp3pm

data.Temp3pm.describe()
plt.figure(figsize=(10,6))

sns.distplot(data_no_nan.Temp3pm)
# Fill with mean, nearly same as median

data.Temp3pm = data.Temp3pm.fillna(data.Temp3pm.mean())
# WindSpeed3pm

data.WindSpeed3pm.describe()
plt.figure(figsize=(10,6))

g = sns.countplot(data_no_nan.WindSpeed3pm)

g.set_xticklabels(g.get_xticklabels(), rotation=90)
# Fill with mean, nearly same as median

data.WindSpeed3pm = data.WindSpeed3pm.fillna(data.WindSpeed3pm.mean())
# Humidity9am

data.Humidity9am.describe()
plt.figure(figsize=(10,6))

g = sns.distplot(data_no_nan.Humidity9am)
# Fill with mean, nearly same as median

data.Humidity9am = data.Humidity9am.fillna(data.Humidity9am.mean())
# Rainfall

data.Rainfall.describe()
plt.figure(figsize=(10,6))

g = sns.distplot(data_no_nan.Rainfall)
# Fill with median

data.Rainfall = data.Rainfall.fillna(data.Rainfall.median())
# WindSpeed9am

data.WindSpeed9am.describe()
plt.figure(figsize=(10,6))

g = sns.countplot(data_no_nan.WindSpeed9am)

g.set_xticklabels(g.get_xticklabels(), rotation=90)
# Fill with mean, nearly same as median

data.WindSpeed9am = data.WindSpeed9am.fillna(data.WindSpeed9am.mean())
# Temp9am

data.Temp9am.describe()
plt.figure(figsize=(10,6))

g = sns.distplot(data_no_nan.Temp9am)
# Fill with mean, nearly same as median

data.Temp9am = data.Temp9am.fillna(data.Temp9am.mean())
# MinTemp

data.MinTemp.describe()
plt.figure(figsize=(10,6))

g = sns.distplot(data_no_nan.MinTemp)
# Fill with mean, nearly same as median

data.MinTemp = data.MinTemp.fillna(data.MinTemp.mean())
# MaxTemp

data.MaxTemp.describe()
plt.figure(figsize=(10,6))

g = sns.distplot(data_no_nan.MaxTemp)
# Fill with mean, nearly same as median

data.MaxTemp = data.MaxTemp.fillna(data.MaxTemp.mean())
# MaxTemp

data.RainToday.value_counts()
plt.figure(figsize=(10,6))

sns.countplot(data_no_nan.RainToday)
# Fill with max value

data.RainToday = data.RainToday.fillna(data.RainToday.value_counts().reset_index().iloc[0]['index'])
(data.isnull().sum() / len(data)).sort_values(ascending=False)
# Converting categorical values to numerical values
# Removing Location and Date, not useful to predict the rain 

data = data.drop(['Date', 'Location'], axis=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(data.WindGustDir)

data.WindGustDir = le.transform(data.WindGustDir)
le = preprocessing.LabelEncoder()

le.fit(data.WindDir9am)

data.WindDir9am = le.transform(data.WindDir9am)
le = preprocessing.LabelEncoder()

le.fit(data.WindDir3pm)

data.WindDir3pm = le.transform(data.WindDir3pm)
data.RainToday = data.RainToday.map({'No':0, 'Yes':1})

data.RainTomorrow = data.RainTomorrow.map({'No':0, 'Yes':1})
data.head()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr())
data.corr()['RainTomorrow'].sort_values(ascending=False)
# RISK_MM is an estimation of the amount of rain in the next day, we don't want to use this feature



data = data.drop('RISK_MM', axis=1)
X = data.drop('RainTomorrow', axis=1)

y = data['RainTomorrow']
from sklearn.model_selection import train_test_split



#split dataset into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier
# Logistic Regression



log = LogisticRegression(random_state=42)

log.fit(X_train,y_train)

precision = 100*round(log.score(X_test, y_test),4)

print('Precision : %s' % precision)
# Random Forest



n_estimators = range(5,100, 5)

precision_rdf = dict()



for i in n_estimators:

    clf = RandomForestClassifier(n_estimators=i,random_state=42)

    clf.fit(X_train, y_train)

    precision = 100*round(clf.score(X_test, y_test),4)

    precision_rdf[i] = precision

    print('Estimators : ', i, '-> Precision : %s' %precision)
best_estimators = max(precision_rdf, key=precision_rdf.get)

clf = RandomForestClassifier(n_estimators=best_estimators,random_state=42)

clf.fit(X_train, y_train)

precision = 100*round(clf.score(X_test, y_test),4)

print(precision)
n_estimators = range(10,150, 10)

precision_xgb = dict()



for i in n_estimators:

    xgb = XGBClassifier(learning_rate=0.1, n_estimators=i, max_depth=8,

                            min_child_weight=3, gamma=0.2, random_state=42)

    xgb.fit(X_train, y_train)

    precision = 100*round(xgb.score(X_test, y_test),4)

    precision_xgb[i] = precision

    print('Estimators : ', i, '-> Precision : %s' %precision)
best_estimator_xgb = max(precision_xgb, key=precision_xgb.get)

xgb = XGBClassifier(learning_rate=0.1, n_estimators=i, max_depth=8,

                            min_child_weight=3, gamma=0.2, random_state=42)

xgb.fit(X_train, y_train)

precision = 100*round(xgb.score(X_test, y_test),4)

print(precision)