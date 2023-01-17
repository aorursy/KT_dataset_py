import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sns.set(font_scale=1.5);
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
plt.style.use("fivethirtyeight")
data_test = '../input/health-diagnostics-test.csv'
data_train = '../input/health-diagnostics-train.csv'
health_test = pd.read_csv(data_test,na_values=('#NULL!'))
health_train = pd.read_csv(data_train,na_values=('#NULL!'))
health_train.describe()
health_train.info()
#setting nulls to 0. They only affect a few rows in the whole dataset, and all independent variables are categorical.
health_train['income'].fillna(0, inplace=True)
health_train['maternal'].fillna(0, inplace=True)
health_train['fam-history'].fillna(0, inplace=True)
health_train['mat-illness-past'].fillna(0, inplace=True)
health_train['suppl'].fillna(0, inplace=True)
health_train['mat-illness'].fillna(0, inplace=True)
health_train['meds'].fillna(0, inplace=True)
health_train['env'].fillna(0, inplace=True)
health_train['lifestyle'].fillna(0, inplace=True)
# Fit a RBF SVM model and store the class predictions.

svc = SVC(random_state=1,probability=True,gamma='auto')

feature_cols = ['income','maternal','fam-history','mat-illness-past','suppl','mat-illness','meds','env','lifestyle']
X = health_train[feature_cols]
y = health_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test) 
y_pred_proba = svc.predict_proba(X_test) 
svc
#Getting a ROC_AUC cross validation score for svc
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

results = model_selection.cross_val_score(svc, X, y, cv=8, scoring='roc_auc')
results.mean()
#Getting the ROC_AUC score with the fitted model.
roc_auc_score(y_test,y_pred_proba[:,1])
#Fit another model with class_weight
svc2 = SVC(class_weight='balanced',random_state=1,gamma='auto',probability=True)
svc2.fit(X_train, y_train)
y_pred = svc2.predict(X_test) 
y_pred_proba = svc2.predict_proba(X_test) 
#Getting a ROC_AUC score for svc2
roc_auc_score(y_test,y_pred_proba[:,1])
#Getting a ROC_AUC cross validation score for svc2 via cross validation
results = model_selection.cross_val_score(svc2, X, y, cv=8, scoring='roc_auc')
results.mean()
#Find the best penalty parameter C for error term
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    svc2 = SVC(class_weight='balanced',C=i,random_state=1,gamma='auto',probability=True)
    svc2.fit(X_train, y_train)
    y_pred_proba = svc2.predict_proba(X_test) 
    print('c = ' , i, ' roc_auc_score = ', roc_auc_score(y_test,y_pred_proba[:,1]))
    

#results show that C=0.1 has a slightly better ROC_AUC
#assigning a smaller gamma
svc3 = SVC(class_weight='balanced',C=0.1,gamma=0.001,random_state=1,probability=True)
svc3.fit(X_train, y_train)
y_pred = svc3.predict(X_test) 
y_pred_proba = svc3.predict_proba(X_test) 
#Getting a ROC_AUC score for svc3
roc_auc_score(y_test,y_pred_proba[:,1])
#Getting a ROC_AUC cross val score for svc3
results = model_selection.cross_val_score(svc3, X, y, cv=8, scoring='roc_auc')
results.mean()
#Results look acceptable and will be used to fit the whole test data set
svc3.fit(X, y)
y_pred = svc3.predict(X) 
y_pred_proba = svc3.predict_proba(X_test) 
#Getting a ROC_AUC score for svc3
roc_auc_score(y_test,y_pred_proba[:,1])
#Result seems to be good enough. svc3 will be used to predict the target for the test dataset
#setting nulls to 0. They only affect a few rows in the whole dataset, and all independent variables are categorical.
health_test['income'].fillna(0, inplace=True)
health_test['maternal'].fillna(0, inplace=True)
health_test['fam-history'].fillna(0, inplace=True)
health_test['mat-illness-past'].fillna(0, inplace=True)
health_test['suppl'].fillna(0, inplace=True)
health_test['mat-illness'].fillna(0, inplace=True)
health_test['meds'].fillna(0, inplace=True)
health_test['env'].fillna(0, inplace=True)
health_test['lifestyle'].fillna(0, inplace=True)
#predicting target for the test data set
health_test_pred=svc3.predict(health_test[feature_cols])
health_test['target']=health_test_pred
health_test['target'].value_counts()
#creating CSV file for submission
#health_test.to_csv('submission_11.csv',columns=['target'],index_label='index')
