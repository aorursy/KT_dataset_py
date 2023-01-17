# Load in our libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import sklearn

#import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import os



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC

from sklearn.model_selection import KFold



sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)



data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
data_train['cell'] = 'Train'

data_test['cell'] = 'Test'

data_test['Survived'] = None

data_train = data_train[['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','cell']]

data_test = data_test[['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','cell']]



data_final = data_train.append(other = data_test)

data_final = data_final.drop(['Name','Ticket'], axis = 1)
# https://medium.com/spikelab/hyperparameter-optimization-using-bayesian-optimization-f1f393dcd36d

# Feature Engineering

def fe_dataset(dataset):

    dataset['has_parch'] = np.where(dataset['Parch']>0, 1, 0)

    dataset['has_sibsp'] = np.where(dataset['Parch']>0, 1, 0)

    dataset['is_adult'] = np.where(dataset['Age']>18, 1, 0)

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    dataset['family_size'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['is_adult'] = np.where(dataset['family_size'] == 1, 1, 0)

    dataset['Cabin'].fillna('Unknown', inplace=True)

    dataset['people_in_charge'] = dataset['Parch'] + dataset['SibSp']

    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

    dataset['Cabin'] = dataset['Cabin'].astype(str).str[0]

    dataset['cabin_embarked'] = dataset['Cabin'] + dataset['Embarked']

    return dataset



def createDummy(dataset): 

    cat_vars=['Cabin','Embarked','cabin_embarked']

    for var in cat_vars:

        print(var)

        cat_list='var'+'_'+var

        cat_list = pd.get_dummies(dataset[var], prefix=var)

        data1=pd.concat([dataset,cat_list], axis = 1)

        dataset=data1

        print(var+' completed')

    

    print('starting 2nd')

    data_vars=dataset.columns.values.tolist()

    to_keep=[i for i in data_vars if i not in cat_vars]

    data_dummy=dataset[to_keep]

    data_dummy.columns.values

    print('finishing 2nd')

    return data_dummy
data_final = fe_dataset(data_final)
data_final = createDummy(data_final)
data_final.to_csv(r'data_final_fe.csv')

data_final_train = data_final[data_final.cell == 'Train'].drop('cell',axis=1)

data_final_test = data_final[data_final.cell == 'Test'].drop('cell',axis=1)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_final_train[data_final_train.columns[1:12]].astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
g = sns.pairplot(data_final_train[data_final_train.columns[1:12]][[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare',

       u'family_size']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
data_final_train.head()

data_final_train['Age'].mean()
# Distribucion de personas que no sobrevivieron

data_final_train[data_final_train.Survived == 0].Age.hist()

plt.title('Histogram of Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('hist_age')
data_final_train['Survived'].value_counts()
sns.countplot(x='Survived', data=data_final_train, palette='hls')

plt.show()
data_final_train.groupby('Survived').mean()
data_final_train.groupby('Pclass').mean()
data_final.head()
X_train = data_final_train.loc[:, data_final_train.columns != 'Survived']

y_train = data_final_train.loc[:, data_final_train.columns == 'Survived']

X_train.drop('PassengerId', axis = 1)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



clf = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial').fit(X_train, y_train.values.ravel().astype('int'))
X_test = data_final_test.loc[:, data_final_test.columns != 'Survived']

y_test = data_final_test.loc[:, data_final_test.columns == 'Survived']



prediction = clf.predict(X_test)
df = pd.DataFrame({'PassengerId': X_test.PassengerId,

                 'Survived': prediction})

df.to_csv(f'gender_submission.csv', index=False)
X_test = data_final_test.loc[:, data_final_train.columns != 'Survived']

X_test = X_test.drop('PassengerId', axis = 1)
vec = pd.DataFrame(rfe.support_)
X_test
# Some useful parameters which will come in handy later on

ntrain = data_final_train.shape[0]

ntest = data_final_test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold( n_splits=NFOLDS, random_state=SEED)

kf = kf.split(ntrain)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None, ):

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

    

# Class to extend XGboost classifer
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

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
# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
kf
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, X_train, y_train, X_test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,X_train, y_train, X_test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, X_train, y_train, X_test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,X_train, y_train, X_test) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc,X_train, y_train, X_test) # Support Vector Classifier



print("Training is complete")
rf_feature = rf.feature_importances(X_train,y_train)

et_feature = et.feature_importances(X_train, y_train)

ada_feature = ada.feature_importances(X_train, y_train)

gb_feature = gb.feature_importances(X_train,y_train)