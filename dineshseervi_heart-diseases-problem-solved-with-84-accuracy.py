#loading_all libraries

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

from scipy import stats

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import csv

import os
warnings.filterwarnings('ignore')

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df_train.head()
df_train.dtypes
#function for missing data

def missing_data(df_train):

    total = df_train.isnull().sum().sort_values(ascending=False)

    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return(missing_data.head(20))
missing_data(df_train)
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corrmat, vmax=.8, square=True);
plt.figure(figsize=(25,20))

sns.factorplot(data=df_train,x='target',y='age',hue='sex')
plt.figure(figsize=(15,10))

sns.relplot(x='trestbps', y='chol', data=df_train,

            kind='line', hue='fbs', col='sex')
plt.figure(figsize=(15,10))

sns.catplot(x='cp',y='oldpeak',data=df_train,hue='target',height=5,aspect=3,kind='box')

plt.title('boxplot')
plt.figure(figsize=(15,15))

sns.relplot(x='restecg', y='thalach', data=df_train,

            kind='line')
plt.figure(figsize=(15,7))

sns.countplot(x='slope',hue='sex',data=df_train,order=df_train['thal'].value_counts().sort_values().index);
sns.set()

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

sns.pairplot(df_train[cols], size = 2.5)

plt.show()
dependent_all=df_train['target']

independent_all=df_train.drop(['target'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(independent_all,dependent_all,test_size=0.3,random_state=100)
log =LogisticRegression()

log.fit(x_train,y_train)
#model on train using all the independent values in df

log_prediction = log.predict(x_train)

log_score= accuracy_score(y_train,log_prediction)

print('Accuracy score on train set using Logistic Regression :',log_score)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, log_prediction)

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_train,log_prediction)

print("AUC on train using Logistic Regression :",metrics.auc(fpr, tpr))

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_train, log_prediction)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

from sklearn.metrics import recall_score

print('recall_score on train set :',recall_score(y_train, log_prediction))

from sklearn.metrics import f1_score

print('F1_sccore on train set :',f1_score(y_train, log_prediction))
#model on train using all the independent values in df

log_prediction = log.predict(x_test)

log_score= accuracy_score(y_test,log_prediction)

print('accuracy score on test using Logisitic Regression :',log_score)
confusion_matrix(y_test, log_prediction)

fpr, tpr, thresholds = metrics.roc_curve(y_test,log_prediction)

print("AUC on test using Logistic Regression :",metrics.auc(fpr, tpr))

average_precision = average_precision_score(y_test, log_prediction)

print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

print('recall_score on test set :',recall_score(y_test, log_prediction))

print('F1_sccore on test set :',f1_score(y_test, log_prediction))
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()

scores = cross_val_score(lr, x_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
xgboost = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.001)
xgboost.fit(x_train,y_train)
#XGBoost model on the train set

XGB_prediction = xgboost.predict(x_train)

XGB_score= accuracy_score(y_train,XGB_prediction)

print('accuracy score on train using XGBoost ',XGB_score)
from sklearn import metrics

print(confusion_matrix(y_train, XGB_prediction))

fpr, tpr, thresholds = metrics.roc_curve(y_train,XGB_prediction)

print("AUC on train using XGBClassifiers:",metrics.auc(fpr, tpr))



average_precision = average_precision_score(y_train, XGB_prediction)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

print('recall_score on train set :',recall_score(y_train, XGB_prediction))

print('F1_sccore on train set :',f1_score(y_train, XGB_prediction))

#XGBoost model on the test

XGB_prediction = xgboost.predict(x_test)

XGB_score= accuracy_score(y_test,XGB_prediction)

print('accuracy score on test using XGBoost :',XGB_score)
from sklearn import metrics

print(confusion_matrix(y_test, XGB_prediction))

fpr, tpr, thresholds = metrics.roc_curve(y_test,XGB_prediction)

print("AUC on test using XGBClassifiers:",metrics.auc(fpr, tpr))



average_precision = average_precision_score(y_test, XGB_prediction)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

print('recall_score on test set :',recall_score(y_test, XGB_prediction))

print('F1_sccore on test set :',f1_score(y_test, XGB_prediction))

xg = xgb.XGBClassifier()

scores = cross_val_score(xg, x_test, y_test, cv=5, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
rfc2=RandomForestClassifier(n_estimators=100)

rfc2.fit(x_train,y_train)
#model on train using all the independent values in df

rfc_prediction = rfc2.predict(x_train)

rfc_score= accuracy_score(y_train,rfc_prediction)

print('accuracy Score on train using RandomForest :',rfc_score)
from sklearn import metrics

print(confusion_matrix(y_train, rfc_prediction))

fpr, tpr, thresholds = metrics.roc_curve(y_train,rfc_prediction)

print("AUC on train using RandomForest :",metrics.auc(fpr, tpr))



average_precision = average_precision_score(y_train, rfc_prediction)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

print('recall_score on train set :',recall_score(y_train, rfc_prediction))

print('F1_sccore on train set :',f1_score(y_train, rfc_prediction))

#model on test using all the indpendent values in df

rfc_prediction = rfc2.predict(x_test)

rfc_score= accuracy_score(y_test,rfc_prediction)

print('accuracy score on test using RandomForest ',rfc_score)


print(confusion_matrix(y_test, rfc_prediction))

fpr, tpr, thresholds = metrics.roc_curve(y_test,rfc_prediction)

print("AUC on test using RandomForest :",metrics.auc(fpr, tpr))



average_precision = average_precision_score(y_test, rfc_prediction)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))

print('recall_score on test set :',recall_score(y_test, rfc_prediction))

print('F1_sccore on test set :',f1_score(y_test, rfc_prediction))

lr = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(lr, x_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
clf = RandomForestClassifier()

grid_values = {'max_features':['auto','sqrt','log2'],'max_depth':[None, 10, 5, 3, 1],

              'min_samples_leaf':[1, 5, 10, 20, 50]}

grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=10, scoring='accuracy')

grid_clf.fit(x_train, y_train)

grid_clf.best_params_

clf = RandomForestClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Training Accuracy :: ', accuracy_score(y_train, clf.predict(x_train)))

print('Test Accuracy :: ', accuracy_score(y_test, y_pred))