# Load libraries

import pandas as pd

import numpy as np

import re

import time

import gc

gc.enable()

import random

import os

import sys

from tqdm import tqdm

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



import sklearn

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import f1_score, roc_auc_score

from sklearn.model_selection import KFold, train_test_split,GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.cluster import KMeans, MiniBatchKMeans



import lightgbm as lgb

from lightgbm.callback import _format_eval_result



from scipy.sparse import vstack, hstack, csr_matrix, save_npz, load_npz



import torch

import torch.nn as nn

import torch.utils.data

from torch.autograd import Variable

import fastai

print("Pytorch version: {}".format(torch.__version__))

print("Fastai version: {}".format(fastai.__version__))

import logging

logging.basicConfig(filename='log.txt',level=logging.DEBUG, format='%(asctime)s %(message)s')
#callback function for LightGBM

def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):

    def _callback(env):

        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:

            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])

            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))

    _callback.order = 10

    return _callback



logger = logging.getLogger('main')

logger.setLevel(logging.DEBUG)

sc = logging.StreamHandler()

logger.addHandler(sc)

fh = logging.FileHandler('hoge.log')

logger.addHandler(fh)



callbacks = [log_evaluation(logger, period=100)]
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train[:5]
# Sample for EDA

print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

# NaN belongs to float

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# full_dataにはtrainとtestのDataframeが格納されており、それぞれに対して処理を行う

full_data = [train, test]



# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    # Create new feature FamilySize as a combination of SibSp and Parch

    # SibSp=兄弟, Parch=両親、つまり兄弟の人数＋両親の人数＋本人(1) = 家族の人数

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize

    # 最初にIsAlone列を全て0で作成し、FamilySizeが1の場合のみ1で上書き

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column

    # Sが最頻値なのでSで埋める

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    # Prepare for a New feature CategoricalAge

    # Age列のNULLSを平均±標準偏差を範囲とするランダムな整数で埋める

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    # Create a new feature Title, containing the titles of passenger names

    dataset['Title'] = dataset['Name'].apply(get_title)

    # Group all non-common titles into one single grouping "Rare"

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Create a new feature CategoricalFare

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

train['CategoricalAge'] = pd.cut(train['Age'], 5)

print(train['CategoricalFare'].unique())

print(train['CategoricalAge'].unique())
for dataset in full_data:

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
# Some useful parameters which will come in handy later on

ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits= NFOLDS, random_state=SEED)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

        return self.clf.fit(x,y).feature_importances_

    

# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_va = x_train[val_index]



        clf.train(x_tr, y_tr)



        oof_train[val_index] = clf.predict(x_va)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

    }



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

    }



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

    }



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

    }



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }



# LightGBM Classifier parameters 

lgb_params = {   

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc',

        'nthread': 4,

        'n_estimators': 500,

        'num_leaves': 2**5-1,

#         'lambda_l1': 0.1,

#         'lambda_l2': 0.1,

#         'bagging_fraction': 0.7, # default = 1.0, 0.0 < bagging_fraction <= 1.0

#         'bagging_freq': 5,

#         'early_stopping_round' = 200

        'learning_rate': 0.1, # default = 0.1

        'min_sum_hessian_in_leaf': 1e-3, # default = 1e-3, min_sum_hessian_in_leaf >= 0.0

        'max_depth': -1,

        }
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

lgbm = SklearnHelper(clf=lgb.LGBMClassifier, seed=SEED, params=svc_params)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel() # flatten label

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

lgb_oof_train, lgb_oof_test = get_oof(lgbm,x_train, y_train, x_test) # LightGBM Classifier



print("Training is complete")
lgb_train = np.zeros((ntrain,))

lgb_test = np.zeros((ntest,))

lgb_test_skf = np.empty((NFOLDS, ntest))



for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):

    x_tr = x_train[train_index]

    y_tr = y_train[train_index]

    x_va = x_train[test_index]

    trn_data = lgb.Dataset(x_tr, label=y_tr)

    lgb_model = lgb.train(lgb_params,trn_data, valid_sets=[trn_data],

                      verbose_eval=0, callbacks=callbacks)



    lgb_train[test_index] = lgb_model.predict(x_va)

    lgb_test_skf[i, :] = lgb_model.predict(x_test)



lgb_test[:] = lgb_test_skf.mean(axis=0)

lgb_oof_train = lgb_train.reshape(-1, 1)

lgb_oof_test = lgb_test.reshape(-1, 1)
rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)

lgb_feature = lgbm.feature_importances(x_train,y_train)

lgb_feature = lgb_feature / sum(lgb_feature)

print(lgb_feature)
cols = train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_feature,

     'Extra Trees  feature importances': et_feature,

      'AdaBoost feature importances': ada_feature,

    'Gradient Boost feature importances': gb_feature,

    'LightGBM feature importances': lgb_feature,                               

    })
# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Extra Trees  feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Extra Trees Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'AdaBoost Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Gradient Boost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')





# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['LightGBM feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['LightGBM feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'LightGBM Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
# Create the new column containing the average of values



feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(3)
y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Barplots of Mean Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

     'GradientBoost': gb_oof_train.ravel(),

     'LightGBM': lgb_oof_train.ravel()

    })

base_predictions_train.head()
data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train, lgb_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test, lgb_oof_test), axis=1)

print(x_train.shape)

print(x_test.shape)
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
# Generate Submission File 

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)