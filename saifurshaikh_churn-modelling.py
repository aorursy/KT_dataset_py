# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
data.head()
data.shape
# Dropping columns that are not required for EDA or modelling
data.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
data.info()
data.head()
data.describe()
sns.countplot(x=data.Geography, data=data, hue=data.Exited)
sns.countplot(x=data.Gender, data=data, hue=data.Exited)
plt.figure(figsize=(12,9))
sns.scatterplot(x=data.EstimatedSalary, y=data.Balance, data=data)
plt.show()
sns.distplot(data.CreditScore)
sns.distplot(data.EstimatedSalary)
sns.distplot(data.Balance)
data['AgeGroup'] = 'Young'
data.loc[(data.Age>35)&(data.Age<=60), 'AgeGroup']='MidAge'
data.loc[data.Age>60, 'AgeGroup']='SeniorCitizens'
data.drop('Age', axis=1, inplace=True)
data.head()
data = pd.get_dummies(data=data, drop_first=True)
data.head()
X = data.drop('Exited', axis=1)
y = data.Exited
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sns.pairplot(data=X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled.head()
y.value_counts()
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
y_train.value_counts()
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred
print(metrics.confusion_matrix(y_test, y_pred))
!pip install lazypredict
!pip install --upgrade pip
import lazypredict
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models
predictions
import imblearn
from imblearn.over_sampling import SMOTE
smt = SMOTE(0.75, random_state=2)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_res, X_test, y_train_res, y_test)
models
predictions
rf = RandomForestClassifier(random_state=2)
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)
y_pred
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(metrics.roc_auc_score(y_test, y_pred))
print(rf.score(X_train_res, y_train_res))
print(rf.score(X_test, y_test))
X_train.shape
y_train.value_counts()
rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight={0:1, 1:5}, random_state=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print('\t')
print(metrics.classification_report(y_test, y_pred))
print('\t')
print(metrics.roc_auc_score(y_test, y_pred))
print('\t')
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate=0.01, max_iter=1000, max_depth=6, validation_fraction=0.2, 
                                     n_iter_no_change=25, max_leaf_nodes=9, min_samples_leaf=20, loss='binary_crossentropy',
                                     l2_regularization=1, random_state=2)
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print('\t')
print(metrics.classification_report(y_test, y_pred))
print('\t')
print(metrics.roc_auc_score(y_test, y_pred))
print('\t')
print(clf.score(X_train_res, y_train_res))
print(clf.score(X_test, y_test))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
X_train.head()
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
import keras
from keras.models import Sequential
from keras.layers import Dense
X_train_res.shape
model = Sequential()
model.add(Dense(32, input_dim = 12, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))

model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))

model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))
history = model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
clf = model.fit(x=X_train_res, y=y_train_res, epochs=100, batch_size=128)
y_pred = model.predict_classes(X_test)
y_pred
print(metrics.confusion_matrix(y_test, y_pred))
print('\t')
print(metrics.classification_report(y_test, y_pred))
print('\t')
print(metrics.roc_auc_score(y_test, y_pred))
