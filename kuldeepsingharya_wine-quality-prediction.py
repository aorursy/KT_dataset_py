# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.info()
data.head()
data.shape
data.describe()
data.isnull().sum()
for col in data.columns:

    print(col)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
fig = plt.figure(figsize = (8,6))

ax = sns.barplot(x="quality", y="fixed acidity", data=data)
fig = plt.figure(figsize = (8,6))

ax = sns.barplot(x="quality", y="volatile acidity", data=data)
fig = plt.figure(figsize=(10,10))

ax = sns.barplot(x='quality', y='citric acid',data=data)
fig = plt.figure(figsize = (8,8))

ax=sns.barplot(x='quality',y='residual sugar',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='chlorides',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='free sulfur dioxide',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='total sulfur dioxide',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='density',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='pH',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='sulphates',data=data)
fig = plt.figure(figsize=(8,8))

ax=sns.barplot(x='quality',y='alcohol',data=data)
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
from sklearn.preprocessing import LabelEncoder

label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'])
data['quality'].value_counts()
sns.countplot(data['quality'])
X = data.drop('quality', axis = 1)

y = data['quality']
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
sgd = SGDClassifier(penalty=None)

sgd.fit(X_train, y_train)

pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))

svc = SVC()

svc.fit(X_train, y_train)

pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))
#Finding best parameters for our SVC model

param = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

}

grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)

grid_svc.fit(X_train, y_train)
grid_svc.best_params_
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')

svc2.fit(X_train, y_train)

pred_svc2 = svc2.predict(X_test)

print(classification_report(y_test, pred_svc2))
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)

rfc_eval.mean()