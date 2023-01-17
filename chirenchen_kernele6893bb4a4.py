# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb1

from xgboost.sklearn import XGBClassifier

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import *

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.tree import *

from sklearn.svm import SVC

from sklearn.model_selection import *

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import *

rnd.seed(6)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

passengerid = test['PassengerId']

train.head(3)
full_data = [train, test]
for data in full_data:

    data['title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

for data in full_data:

    data['title'] = data['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 

                                                'Rare')

    data['title'] = data['title'].replace('Mlle', 'Miss')

    data['title'] = data['title'].replace('Ms', 'Miss')

    data['title'] = data['title'].replace('Mme', 'Mrs')

train[['Survived', 'title']].groupby('title', as_index=False).mean().sort_values(by='Survived',ascending=False)
for data in full_data:

    data['title'] = data['title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})

train.head()
train.describe()
for data in full_data:

    data['Sex'] = data['Sex'].replace('female', int(1))

    data['Sex'] = data['Sex'].replace('male', int(0))

    data['Sex'] = data['Sex'].astype(int)

train.head()
ageguess = np.zeros((2,3))

for data in full_data:

    for i in range(2):

        for j in range(3):

            agemean =data.loc[(data["Sex"] == i) & (data['Pclass'] == j+1), 'Age'].mean()

            agestd = data.loc[(data["Sex"] == i) & (data['Pclass'] == j+1), 'Age'].std()

            age_guess = rnd.uniform(agemean-agestd, agemean+agestd)

            ageguess[i][j] = int(age_guess/0.5 + 0.5) * 0.5

    for i in range(2):

        for j in range(3):

            data.loc[(data["Sex"] == i) & (data['Pclass'] == j+1) & (data['Age'].isnull()), 'Age'] = ageguess[i][j]

    data['Age'] = data['Age'].astype(int)

            

            
for data in full_data:

    data['ageband'] = pd.cut(data['Age'], 5)

train[['Survived', 'ageband']].groupby('ageband', as_index=False).mean().sort_values(by='ageband', ascending=False)
for data in full_data:

    data.loc[data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[data['Age'] > 64, 'Age'] = 4

train.head(5)

    
for data in full_data:

    data['familysize'] = data['SibSp'] + data['Parch'] +1

for data in full_data:

    data['isalone'] = 0

    data.loc[data['familysize'] == 1, 'isalone'] = 1

for data in full_data:

    data['age*pclass'] = data['Age'] * data['Pclass']
for data in full_data:

    data['Embarked'] = data['Embarked'].fillna('S')

for data in full_data:

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

for data in full_data:

    data['fareband'] = pd.qcut(data['Fare'], 4)

train[['Survived', 'fareband']].groupby('fareband', as_index=False).mean().sort_values(by='fareband', ascending=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
for data in full_data:

    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2

    data.loc[(data['Fare'] > 31), 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'familysize','SibSp', 'Parch']

train = train.drop(drop_elements, axis=1)

train = train.drop(['ageband', 'fareband'], axis=1)

test = test.drop(drop_elements, axis=1)

test = test.drop(['ageband', 'fareband'], axis=1)
train.head()
test.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train, hue='Survived', palette='seismic', size=1.2, diag_kind='kde', diag_kws=dict(shade=True),

                plot_kws=dict(s=10))

g.set(xticklabels=[])
mtrain = train.shape[0]

mtest = test.shape[0]

seed = 0

nfolds =5

kf = KFold( n_splits=nfolds, random_state=seed)

class sklearnhelper(object):

    def __init__(self,clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)

    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)

    def predict(self, x):

        return self.clf.predict_proba(x)

    def fit(self, x, y):

        return self.clf.fit(x, y)

    def feature_importances(self, x, y):

        return self.clf.fit(x, y).feature_importances_

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((mtrain, ))

    oof_test = np.zeros((mtest, ))

    oof_test_skf = np.zeros((nfolds, mtest))

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)[:, 1]

        oof_test_skf[i, :] = clf.predict(x_test)[:, 1]

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    

rfparams = {

    'n_jobs': -1,

    'n_estimators': 575,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',



}

etparams = {

    'n_jobs': -1,

    'n_estimators':575,

    #'max_features': 0.5,

    'max_depth': 5,

    'min_samples_leaf': 3,



}

adaparams = {

    'n_estimators': 575,

    'learning_rate' : 0.95

}



gbparams = {

    'n_estimators': 575,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 3,



}

svcparams = {

    'kernel' : 'linear',

    'C' : 0.025,

 

            'probability': True}

rf = sklearnhelper(clf=RandomForestClassifier, seed=seed, params=rfparams)

et = sklearnhelper(clf=ExtraTreesClassifier, seed=seed, params=etparams)

ada =sklearnhelper(clf=AdaBoostClassifier, seed=seed, params=adaparams)

gb = sklearnhelper(clf=GradientBoostingClassifier, seed=seed, params=gbparams)

svc = sklearnhelper(clf=SVC, seed=seed, params=svcparams)

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

x_train = train.values

x_test = test.values

logparams = {'n_jobs': -1}

xgbparams = {'n_estimator': 1000, 'max_depth': 6, 'gamma': 0.9, 'subsample': 0.8,

            'colsample_bytree': 0.8}

dtparams = {}

bagparams = {}
log = sklearnhelper(clf=LogisticRegression, seed=seed, params=logparams)

bag = sklearnhelper(clf=BaggingClassifier, seed=seed, params=bagparams) 

xgb = sklearnhelper(clf=XGBClassifier, seed=seed, params=xgbparams) 

dt = sklearnhelper(clf=DecisionTreeClassifier, seed=seed, params=dtparams) 
class helper(object):

    def __init__(self,clf, params=None):

        self.clf = clf(**params)

    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)

    def predict(self, x):

        return self.clf.predict(x)

    def fit(self, x, y):

        return self.clf.fit(x, y)

def get_helper(clf, x_train, y_train, x_test):

    oof_train = np.zeros((mtrain, ))

    oof_test = np.zeros((mtest, ))

    oof_test_skf = np.zeros((nfolds, mtest))

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    

knnparams = {}





knn = helper(clf=KNeighborsClassifier,  params=knnparams)



et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)

rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)

gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)

svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)

log_oof_train, log_oof_test = get_oof(log, x_train, y_train, x_test)

bag_oof_train, bag_oof_test = get_oof(bag, x_train, y_train, x_test)

dt_oof_train, dt_oof_test = get_oof(dt, x_train, y_train, x_test)

xgb_oof_train, xgb_oof_test = get_oof(xgb, x_train, y_train, x_test)

knn_oof_train, knn_oof_test = get_helper(knn, x_train, y_train, x_test)

def accuracy(trainproba, ytrain):

    trainproba = (trainproba>0.5)

    ans = np.sum(np.abs(trainproba - ytrain.reshape(891, 1))) / 891

    return (1- ans)*100
print('''

et_oof_train : {}

rf_oof_train : {}

ada_oof_train : {}

gb_oof_train : {}

svc_oof_train : {}

log_oof_train : {}

bag_oof_train : {}

dt_oof_train : {}

xgb_oof_train : {}

knn_oof_train : {}



'''.format(accuracy(et_oof_train, y_train), accuracy(rf_oof_train, y_train), accuracy(ada_oof_train, y_train)

          ,accuracy(gb_oof_train, y_train) ,accuracy(svc_oof_train, y_train), accuracy(log_oof_train, y_train),

          accuracy(bag_oof_train, y_train), accuracy(dt_oof_train, y_train),

          accuracy(xgb_oof_train, y_train), accuracy(knn_oof_train, y_train)))
rffeature = rf.feature_importances(x_train, y_train)

etfeature = et.feature_importances(x_train, y_train)

adafeature = ada.feature_importances(x_train, y_train)

gbfeature = gb.feature_importances(x_train, y_train)
cols = train.columns.values

feature_importances_dataframe = pd.DataFrame({'features': cols,

                                              'random forest': rffeature,

                                             'extra trees': etfeature,

                                             'adaboost': adafeature,

                                             'gradient boost': gbfeature})

feature_importances_dataframe
feature_importances_dataframe['features'].values,
trace = go.Scatter(

    y = feature_importances_dataframe['random forest'].values,

    x = feature_importances_dataframe['features'].values,

    mode = 'markers+lines',

    marker = dict(

             sizemode = 'diameter',

             sizeref = 1,

             size = 25,

             color = feature_importances_dataframe['random forest'].values,

             colorscale = 'Portland',

             showscale = True

    ),

    text = feature_importances_dataframe['features'].values

)

data = [trace]

layout = go.Layout(

    autosize = True,

    title = 'random forest feature importantce',

    hovermode = 'closest',

    yaxis = dict(

        title = 'feature importance',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='scatter2010')
trace = go.Scatter(

    y = feature_importances_dataframe['extra trees'].values,

    x = feature_importances_dataframe['features'].values,

    mode = 'markers',

    marker = dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_importances_dataframe['extra trees'].values,

        colorscale = 'Portland',

        showscale = True

    ),

    text = feature_importances_dataframe['features'].values

)



data = [trace]

layout = go.Layout(

    autosize = True,

    title = 'extra trees feature importance',

    hovermode = 'closest',

    yaxis = dict(

        title = 'feature importance',

        ticklen = 5,

        gridwidth = 2,

    ),

    showlegend = False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='scatter2010')
trace = go.Scatter(

    y = feature_importances_dataframe['adaboost'].values,

    x = feature_importances_dataframe['features'].values,

    mode = 'markers',

    marker = dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_importances_dataframe['adaboost'].values,

        colorscale = 'Portland',

        showscale = True

        )

)

data = [trace]

layout = go.Layout(

    autosize = True,

    title = 'adaboost',

    hovermode = 'closest',

    yaxis = dict(

        title = 'feature importance',

        ticklen = 5,

        gridwidth = 2,

    ),

    showlegend = False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='scatter2010')

trace = go.Scatter(

    y = feature_importances_dataframe['gradient boost'].values,

    x = feature_importances_dataframe['features'].values,

    mode = 'markers',

    marker = dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

        color = feature_importances_dataframe['gradient boost'].values,

        colorscale = 'Portland',

        showscale = True),

)

data = [trace]

layout = go.Layout(

    autosize = True,

    title = 'gradient bost',

    hovermode = 'closest',

    yaxis = dict(

        title = 'feature importances',

        ticklen = 5,

        gridwidth = 2

    ),

    showlegend = False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig , filename='scatter2010')
feature_importances_dataframe['mean'] = feature_importances_dataframe.mean(axis=1)

trace = go.Bar(

    y = feature_importances_dataframe['mean'].values,

    x = feature_importances_dataframe['features'].values,

    width = 0.5,

    marker = dict(

        color = feature_importances_dataframe['mean'].values,

        colorscale = 'Portland',

        showscale = True,

        reversescale = False

    ),

    opacity = 0.6

)

data = [trace]

layout = go.Layout(

    autosize =True,

    title = 'Mean feature_importances_dataframe',

    hovermode = 'closest',

    yaxis = dict(

        title = 'feature_importances_dataframe',

        ticklen = 5,

        gridwidth  = 2

    ),

    showlegend = False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar_direct_label')
base_predict_train = pd.DataFrame(

    {'random_a': rf_oof_train.ravel(),



     'extratrees_a': et_oof_train.ravel(),



     'adaboost_a': ada_oof_train.ravel(),



     'gradientboost_a': gb_oof_train.ravel(),

})

base_predict_train.head(5)
data = [go.Heatmap(

    z = base_predict_train.astype(float).corr().values,

    x = base_predict_train.columns.values,

    y = base_predict_train.columns.values,

    colorscale = 'Viridis',

    showscale = True,

    reversescale = True)]

py.iplot(data, filename='labeled-heatmap')
x_train = np.concatenate((x_train, rf_oof_train, et_oof_train, gb_oof_train, ada_oof_train, svc_oof_train,

                         log_oof_train, bag_oof_train, dt_oof_train, xgb_oof_train), axis=1 )

x_test = np.concatenate((x_test, rf_oof_test, et_oof_test, gb_oof_test, ada_oof_test, svc_oof_test,

                        log_oof_test, bag_oof_test, dt_oof_test, xgb_oof_test), axis=1 )



x_test.shape
mtrain = train.shape[0]

mtest = test.shape[0]

seed = 0

nfolds =5

kf = KFold( n_splits=nfolds, random_state=seed)

class sklearnhelper(object):

    def __init__(self,clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)

    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)

    def predict(self, x):

        return self.clf.predict_proba(x)

    def fit(self, x, y):

        return self.clf.fit(x, y)

    def feature_importances(self, x, y):

        return self.clf.fit(x, y).feature_importances_

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((mtrain, ))

    oof_test = np.zeros((mtest, ))

    oof_test_skf = np.zeros((nfolds, mtest))

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)[:, 1]

        oof_test_skf[i, :] = clf.predict(x_test)[:, 1]

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    
rfparams = {

    'n_jobs': -1,

    'n_estimators': 575,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt'}

etparams = {

    'n_jobs': -1,

    'n_estimators':575,

    #'max_features': 0.5,

    'max_depth': 5,

    'min_samples_leaf': 3}

gbparams = {

    'n_estimators': 575,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 3}

svcparams = {

    'kernel' : 'linear',

    'C' : 0.025,

 'probability': True}

xgbparams = {'n_estimator': 1000, 'max_depth': 6, 'gamma': 0.9, 'subsample': 0.8,

            'colsample_bytree': 0.8}

rf = sklearnhelper(clf=RandomForestClassifier, seed=seed, params=rfparams)

et = sklearnhelper(clf=ExtraTreesClassifier, seed=seed, params=etparams)

xgb = sklearnhelper(clf=XGBClassifier, seed=seed, params=xgbparams) 

gb = sklearnhelper(clf=GradientBoostingClassifier, seed=seed, params=gbparams)

svc = sklearnhelper(clf=SVC, seed=seed, params=svcparams)
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)

rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)

gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)

svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)

xgb_oof_train, xgb_oof_test = get_oof(xgb, x_train, y_train, x_test)
x_train = np.concatenate((rf_oof_train, et_oof_train, gb_oof_train,  svc_oof_train,

                         xgb_oof_train), axis=1 )

x_test = np.concatenate((rf_oof_test, et_oof_test, gb_oof_test,  svc_oof_test,

                        xgb_oof_test), axis=1 )

x_train.shape
gbm = xgb1.XGBClassifier(

learning_rate = 0.95,

 n_estimators= 5000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=1,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

pred = gbm.predict(x_test)
StackingSubmission = pd.DataFrame({'PassengerId': passengerid, 'Survived': pred})

StackingSubmission.to_csv('StackingSubmission.csv', index=False)