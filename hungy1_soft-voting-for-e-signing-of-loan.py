#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', None)

%matplotlib inline
#importing data

data = pd.read_csv("../input/esigning-of-loan-based-on-financial-history/financial_data.csv")

print(data.shape)

data.head()
#checking for imbalanced data

data.e_signed.value_counts()
#ploting the corolation matrix to detect the multicolinearity



plt.figure(figsize=(22,12))

sns.heatmap(data.corr(), annot =True, cmap='viridis')
#split the data into train and test sets

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data.drop('entry_id', axis =1), data.e_signed, test_size =0.2, random_state =0)
#find categorical features

cat_var = [col for col in X_train.columns if X_train[col].dtype =='O']

cat_var
#checking for null values in train and test sets

[col for col in X_train.columns if X_train[col].isnull().sum() > 0]
[col for col in X_test.columns if X_test[col].isnull().sum() > 0]


def categorical_encode(var , target):

    order = X_train.groupby(var)[target].mean().to_dict()

    X_train[var] = X_train[var].map(order)

    X_test[var] = X_test[var].map(order)
categorical_encode('pay_schedule', 'e_signed')
X_train = X_train.drop('e_signed', axis =1)

X_test= X_test.drop('e_signed', axis =1)
X_train.describe()
#standarization of train and test set

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train_scalled = sc.fit_transform(X_train)

X_test_scaled = sc.transform(X_test)
from sklearn.metrics import roc_auc_score
import xgboost as xgb

xgb_model  = xgb.XGBClassifier()



xgb_model.fit(X_train, y_train)



pred = xgb_model.predict_proba(X_train)

print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))



pred = xgb_model.predict_proba(X_test)

xgb_pred=roc_auc_score(y_test, pred[:,1])

print('xgb test roc-auc: {}'.format(xgb_pred))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200)

rf_model.fit(X_train, y_train)



pred = rf_model.predict_proba(X_train)

print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = rf_model.predict_proba(X_test)

rf_pred=roc_auc_score(y_test, pred[:,1])

print('RF test roc-auc: {}'.format(rf_pred))

#clear sign of overcfitting
##takes time

from sklearn.model_selection import GridSearchCV

params={ 'max_depth':[6,10],

        'criterion':['gini', 'entropy'],

        'max_leaf_nodes':[5,8,10]

}

gridsearch = GridSearchCV(rf_model, params, scoring ='accuracy', cv =10)

gridsearch = gridsearch.fit(X_train, y_train)
gridsearch.best_params_
rf_model = RandomForestClassifier(n_estimators=200, criterion = 'gini', max_depth = 10, max_leaf_nodes = 10)

rf_model.fit(X_train, y_train)



pred = rf_model.predict_proba(X_train)

print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = rf_model.predict_proba(X_test)

rf_pred=roc_auc_score(y_test, pred[:,1])

print('RF test roc-auc: {}'.format(rf_pred))
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier()

ada_model.fit(X_train, y_train)



pred = ada_model.predict_proba(X_train)

print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = ada_model.predict_proba(X_test)

ada_pred =roc_auc_score(y_test, pred[:,1])

print('Adaboost test roc-auc: {}'.format(ada_pred))
from sklearn.svm import SVC

svc = SVC(probability=True)

svc.fit(X_train_scalled, y_train)

pred = svc.predict_proba(X_train_scalled)

print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = svc.predict_proba(X_test_scaled)

svc_pred =roc_auc_score(y_test, pred[:,1])

print('RF test roc-auc: {}'.format(svc_pred))
models = pd.DataFrame({'Models':['xgboost','random forest','adaboost', 'svc'],

                     'score' :[xgb_pred,rf_pred,ada_pred, svc_pred]})

models
models.sort_values(by ='score', ascending =False)

from sklearn.ensemble import VotingClassifier



estimators = [('xgb' ,xgb_model) , ('rf', rf_model),('ada', ada_model ),('svc', svc)]

voting_cl = VotingClassifier(estimators, voting='soft')

              
ensemble = voting_cl.fit(X_train, y_train)
ensemble_prediction = ensemble.predict_proba(X_test)[:, 1]
ensemble_prediction
roc_auc_score(y_test, ensemble_prediction)