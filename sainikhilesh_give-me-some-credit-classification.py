# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#reading the data
sampleEntry = pd.read_csv('../input/credit/sampleEntry.csv')
train = pd.read_csv('../input/credit/cs-training.csv')
test = pd.read_csv('../input/credit/cs-test.csv')
#dimension of the data
print(train.shape)
print(test.shape)
#getting first five observations
train.head()
#getting first five observations 
test.head()
#describing train data
train.describe()
print(train.isnull().sum())
print(test.isnull().sum())
train.nunique()
train['MonthlyIncome'].fillna(train['MonthlyIncome'].mean(),inplace=True)
train['NumberOfDependents'].fillna(train['NumberOfDependents'].mode()[0], inplace=True)
test['MonthlyIncome'].fillna(test['MonthlyIncome'].mean(),inplace=True)
test['NumberOfDependents'].fillna(test['NumberOfDependents'].mode()[0], inplace=True)
print(train.isnull().sum())
print(test.isnull().sum())
#plot two tyep classe "0" and "1"
sns.countplot(x='SeriousDlqin2yrs',data=train)
plt.show()
cor=train.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,annot=True,ax=ax)
X = train.drop('SeriousDlqin2yrs',1)
y = train['SeriousDlqin2yrs']
train.columns
from sklearn.model_selection import train_test_split
#splitting data into train and test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=568)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
xgb = XGBClassifier(n_jobs=-1) 
 
# Use a grid over parameters of interest
param_grid = {
                  'n_estimators' :[100,150,200,250,300],
                  "learning_rate" : [0.001,0.01,0.0001,0.05, 0.10 ],
                  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3 ],
                  "colsample_bytree" : [0.5,0.7],
                  'max_depth': [3,4,6,8]
              }
xgb_randomgrid = RandomizedSearchCV(xgb, param_distributions=param_grid, cv=5)
%%time
xgb_randomgrid.fit(X_train,y_train)
best_est = xgb_randomgrid.best_estimator_
y_pred = best_est.predict_proba(X_train)
y_pred = y_pred[:,1]
from sklearn.metrics import auc,roc_curve
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
X_test = test.drop('SeriousDlqin2yrs',1)
y_test=best_est.predict_proba(X_test)
y_test= y_test[:,1]
print(y_test)
sampleEntry["Probability"]=y_test
sampleEntry.head()
sampleEntry.to_csv("submission.csv",index=False)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)

os_data_X,os_data_y=smote.fit_sample(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt') 
 
# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [9, 18, 27, 36, 100, 150],
           "max_depth" : [2,3,5,7,9],
           "min_samples_leaf" : [2, 4]}
rfc_randomgrid = RandomizedSearchCV(rfc, param_distributions=param_grid, cv=5)
rfc_randomgrid.fit(os_data_X,os_data_y)
best_est1 = rfc_randomgrid.best_estimator_
y_pred1 = best_est1.predict_proba(X_train)
y_pred1 = y_pred1[:,1]
y_test1=best_est1.predict_proba(X_test)
y_test1= y_test1[:,1]
fpr,tpr,_ = roc_curve(y_train, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
sampleEntry["Probability"]=y_test1
sampleEntry.head()
sampleEntry.to_csv("submission1.csv",index=False)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred2 = knn.predict_proba(X_train)
y_pred2 = y_pred2[:,1]
fpr,tpr,_ = roc_curve(y_train, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
md_KNN = KNeighborsClassifier()

neighbors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
param_grid = dict(n_neighbors=neighbors)
KNN_GridSearch = GridSearchCV(md_KNN,param_grid=param_grid,cv=10)
KNN_GridSearch.fit(X_train,y_train)
best_est2=KNN_GridSearch.best_estimator_
y_pred3 = best_est2.predict_proba(X_train)
y_pred3 = y_pred3[:,1]
fpr,tpr,_ = roc_curve(y_train, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,8))
plt.title('Receiver Operating Characteristic')
sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
y_test2=best_est2.predict_proba(X_test)
y_test2= y_test2[:,1]
sampleEntry["Probability"]=y_test2
sampleEntry.head()
sampleEntry.to_csv("submission2.csv",index=False)
