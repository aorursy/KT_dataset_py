import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from scipy.stats import norm 

from matplotlib import cm

import seaborn as sns
df_train = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv')

df_test = pd.read_csv('../input/GiveMeSomeCredit/cs-test.csv')

df_s = pd.read_csv('../input/GiveMeSomeCredit/sampleEntry.csv')
df_train.head()
df_test.head()
print(df_train.shape)

print(df_test.shape)
df_train['Id'] = df_train['Unnamed: 0']
df_train.drop('Unnamed: 0', axis=1, inplace=True)
df_train.head()
df_test['Id'] = df_test['Unnamed: 0']
df_test.drop('Unnamed: 0', axis=1, inplace=True)
df_test.head()
df_train.describe()
df_train.isnull().sum()
df_train.nunique()
df_test.isnull().sum()
df_train['MonthlyIncome'].fillna(df_train['MonthlyIncome'].mean(),inplace=True)
df_train['NumberOfDependents'].fillna(df_train['NumberOfDependents'].mode()[0], inplace=True)
df_test['MonthlyIncome'].fillna(df_test['MonthlyIncome'].mean(),inplace=True)
df_test['NumberOfDependents'].fillna(df_test['NumberOfDependents'].mode()[0], inplace=True)
df_train.isnull().sum()
df_test.isnull().sum()
sns.countplot(x='SeriousDlqin2yrs',data=df_train)

plt.show()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.show()
Id = df_test['Id']
df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)
X = df_train.drop('SeriousDlqin2yrs',axis=1)

y = df_train['SeriousDlqin2yrs']
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
X_train, X_test, y_train, y_test = train_test_split( X.values, y.values, test_size=0.2, random_state=116214 )
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

RF = RandomForestClassifier()
param_grid = { 

           "n_estimators" : [9, 18, 27, 36, 100, 150],

           "max_depth" : [2,3,5,7,9],

           "min_samples_leaf" : [2, 4, 6, 8]}
RF_random = RandomizedSearchCV(RF, param_distributions=param_grid, cv=5)
RF_random.fit(X_train, y_train)
best_est_RF = RF_random.best_estimator_
print('Accuracy of classifier on training set: {:.2f}'.format(RF_random.score(X_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(RF_random.score(X_test, y_test) * 100))
y_pred = best_est_RF.predict_proba(X_train)

y_pred = y_pred[:,1]
from sklearn.metrics import roc_curve, auc
fpr,tpr,_ = roc_curve(y_train, y_pred)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,8))

plt.title('Receiver Operating Characteristic')

sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
df_test.drop('SeriousDlqin2yrs', axis=1, inplace=True)

y_pred=best_est_RF.predict_proba(df_test)

y_pred= y_pred[:,1]
df_s["Probability"]=y_pred

df_s.head()
df_s.to_csv("submission_RF.csv",index=False)
XGB = XGBClassifier(n_jobs=-1) 

 

param_grid = {

                  'n_estimators' :[100,150,200,250,300],

                  "learning_rate" : [0.001,0.01,0.0001,0.05, 0.10 ],

                  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3 ],

                  "colsample_bytree" : [0.5,0.7],

                  'max_depth': [3,4,6,8]

              }
XGB_random = RandomizedSearchCV(XGB, param_distributions=param_grid, cv=5)
XGB_random.fit(X_train,y_train)
best_est_XGB = XGB_random.best_estimator_
print('Accuracy of classifier on training set: {:.2f}'.format(XGB_random.score(X_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(XGB_random.score(X_test, y_test) * 100))
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)



s_X,s_y=smote.fit_sample(X_train,y_train)
RF_random.fit(s_X,s_y)
best_est_RF1 = RF_random.best_estimator_
print('Accuracy of classifier on training set: {:.2f}'.format(RF_random.score(s_X,s_y) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(RF_random.score(s_X,s_y) * 100))
y_pred_RF1 = best_est_RF1.predict_proba(X_train)

y_pred_RF1 = y_pred_RF1[:,1]
fpr,tpr,_ = roc_curve(y_train, y_pred_RF1)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,8))

plt.title('Receiver Operating Characteristic')

sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
prediction_RF = best_est_RF1.predict_proba(df_test)

prediction_RF = prediction_RF[:,1]
df_s["Probability"]=prediction_RF

df_s.head()
df_s.to_csv("submission_RF_S.csv",index=False)