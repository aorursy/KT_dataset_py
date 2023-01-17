# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import sklearn

import xgboost as xgb

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,roc_curve,mean_absolute_error

import scipy.stats as stats

import plotly.offline as py

py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plot 

%matplotlib inline

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import KFold

#k折交叉验证函数

import pylab

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId=test['PassengerId']

train.head()
full_data=[train,test]

train['name_length']=train['Name'].apply(len)

test['name_length']=test['Name'].apply(len)

train['has_cabin']=train["Cabin"].apply(lambda x:0 if type(x)==float else 1)

test['has_cabin']=test["Cabin"].apply(lambda x:0 if type(x)==float else 1)

for dataline in full_data:

    dataline["familysize"]=dataline['SibSp']+dataline['Parch']+1

for dataline in full_data:

    dataline["isalone"]=0

    dataline.loc[dataline['familysize']==1,'isalone']=1

    dataline['Embarked']= dataline['Embarked'].fillna('S')

    dataline['Fare']=dataline['Fare'].fillna(train['Fare'].median())

train['categoricalfare']=pd.cut(train['Fare'],4)

for dataset in full_data:

    age_avg=dataset['Age'].mean()

    age_std=dataset['Age'].std()

    age_null_count=dataset['Age'].isnull().sum()

    age_null_random_list=np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list

    dataset['Age']=dataset['Age'].astype(int)

train['categoricalage']=pd.cut(train['Age'],5)

def get_title(name):

    title_search=re.search('([A-Za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    return ""

for dataset in full_data:

    dataset['Title']=dataset['Name'].apply(get_title)

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train.head()

for dataset in full_data:

    dataset['Sex']=dataset['Sex'].map({'female':0,'male':1}).astype(int)

    title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}

    dataset['Title']=dataset['Title'].map(title_mapping)

    dataset['Title']=dataset['Title'].fillna(0)

    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2})

    dataset.loc[dataset['Fare']<=7.91,'Fare']=0

    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare']=1

    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare']=2

    dataset.loc[dataset['Fare']>31,'Fare']=3

    dataset['Fare']=dataset['Fare'].astype(int)

    dataset.loc[dataset['Age']<=16,'Age']=0

    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1

    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2

    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3

    dataset.loc[dataset['Age']>64,'Age']=4

train.head()
title=train['Title'].ravel()

drop_elements=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train=train.drop(drop_elements,axis=1)

train=train.drop(['categoricalage', 'categoricalfare'],axis=1)

test=test.drop(drop_elements,axis=1)

train.head(10)

colormap=plot.cm.RdBu

plot.figure(figsize=(14,12))

plot.title('preson correlation of features',y=1.05,size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)



g=sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

       u'familysize', u'Title']],hue='Survived',palette='seismic',size=1.2,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))

g.set(xticklabels=[])
ntrain=train.shape[0]

ntest=test.shape[0]

SEED=0

nfolds=5

kf=KFold(n_splits=nfolds,random_state=SEED)



class SKlearnHelper(object):

    def __init__(self,clf,seed=0,params=None):

#         params['random_state']=seed

        self.clf=clf(**params)

    def train(self,x_train,y_train):

        self.clf.fit(x_train,y_train)

    def predict(self,x):

        return self.clf.predict(x)

    def fit(self,x,y):

        return self.clf.fit(x,y)

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)



# output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})

# output.to_csv("my_submission2.csv",index=False)

# print("successfully saved!")
def get_oof(clf,x_train,y_train,x_test):

    oof_train=np.zeros((ntrain,))

    oof_test=np.zeros((ntest,))

    oof_test_skf=np.empty((nfolds,ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr=x_train[train_index]

        y_tr=y_train[train_index]

        x_te=x_train[test_index]

        clf.train(x_tr,y_tr)

        oof_train[test_index]=clf.predict(x_te)

        oof_test_skf[i,:]=clf.predict(x_test)

    oof_test[:]=oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

    

        
rf_params={

    'n_jobs':-1,

    'n_estimators':500,

    'warm_start':True,

    'max_depth':6,

    'min_samples_leaf':2,

    'max_features':'sqrt',

    'verbose':0

}

et_params={

    'n_jobs':-1,

    'n_estimators':500,

    'max_depth':8,

    'min_samples_leaf':2,

    'verbose':0

}

ada_params={

    'n_estimators':500,

    'learning_rate':0.75

}

gb_params={

    'n_estimators':500,

    'max_depth':8,

    'min_samples_leaf':2,

    'verbose':0

}

svc_params={

    'kernel':'linear',

    'C':0.025

}

knn_params={

    'n_neighbors':3

}

dst_params={

    'max_depth':8

}
rf=SKlearnHelper(clf=RandomForestClassifier,seed=SEED,params=rf_params)

et=SKlearnHelper(clf=ExtraTreesClassifier,seed=SEED,params=et_params)

ada=SKlearnHelper(clf=AdaBoostClassifier,seed=SEED,params=ada_params)

gb=SKlearnHelper(clf=GradientBoostingClassifier,seed=SEED,params=gb_params)

svc=SKlearnHelper(clf=SVC,seed=SEED,params=svc_params)

knn=SKlearnHelper(clf=KNeighborsClassifier,seed=SEED,params=knn_params)

dst=SKlearnHelper(clf=DecisionTreeClassifier,seed=SEED,params=dst_params)
y_train=train['Survived'].ravel()

train=train.drop(['Survived'],axis=1)

x_train=train.values

x_test=test.values
et_oof_train,et_oof_test=get_oof(et,x_train,y_train,x_test)

rf_oof_train,rf_oof_test=get_oof(rf,x_train,y_train,x_test)

ada_oof_train,ada_oof_test=get_oof(ada,x_train,y_train,x_test)

gb_oof_train,gb_oof_test=get_oof(gb,x_train,y_train,x_test)

svc_oof_train,svc_oof_test=get_oof(svc,x_train,y_train,x_test)

knn_oof_train,knn_oof_test=get_oof(knn,x_train,y_train,x_test)

dst_oof_train,det_oof_test=get_oof(dst,x_train,y_train,x_test)

print("training is complete")
base_prediction_train=pd.DataFrame({'randomforest':rf_oof_train.ravel(),

    'extratrees':et_oof_train.ravel(),

    'adaboost':ada_oof_train.ravel(),

    'gradientboost':gb_oof_train.ravel(),                                

#     'svc':svc_oof_train().ravel()

})

base_prediction_train.head()
x_train=np.concatenate((et_oof_train,rf_oof_train,gb_oof_train,ada_oof_train,svc_oof_train,knn_oof_train,dst_oof_train),axis=1)

x_test=np.concatenate((et_oof_test,rf_oof_test,gb_oof_test,ada_oof_test,svc_oof_test,knn_oof_test,det_oof_test),axis=1)

gbm=xgb.XGBClassifier(

n_estimators=2000,

    max_depth=4,

    min_child_weight=2,

    gamma=0.9,

    subsample=0.8,

    colsample_bytree=0.8,

    objective='binary:logistic',

    nthread=-1,

    scale_pos_weight=1).fit(x_train,y_train)

predictions=gbm.predict(x_test)

stackingsubmission=pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions})

stackingsubmission.to_csv("stackingsubmission.csv",index=False)