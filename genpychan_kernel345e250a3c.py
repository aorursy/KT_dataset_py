# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read dataset

train = pd.read_csv('../input/bike-sharing-demand/train.csv')

test = pd.read_csv('../input/bike-sharing-demand/test.csv')

train.head()
test.head()
sns.factorplot(x='season', data=train, kind='count',size=5, aspect=1.5)
sns.factorplot(x='holiday', data=train, kind='count', size=5)
sns.factorplot(x='workingday', data=train, kind='count', size=5)
sns.factorplot(x='weather', data=train, kind='count',size=5, aspect=1.5)
train['weather'].value_counts()
train.describe()
sns.boxplot(data=train[['temp','atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])

fig = plt.gcf()

fig.set_size_inches(10,8)
train.temp.hist()
season=pd.get_dummies(train['season'],prefix='season')

train = pd.concat([train,season], axis=1)

train.head()
season=pd.get_dummies(test['season'],prefix='season')

test=pd.concat([test,season],axis=1)

test.head()
weather = pd.get_dummies(train['weather'], prefix='weather')

train = pd.concat([train, weather], axis=1)

train.head()
weather = pd.get_dummies(test['weather'], prefix='weather')

test = pd.concat([test, weather], axis=1)

test.head()
train.drop(['season', 'weather'], inplace=True, axis=1)

test.drop(['season', 'weather'], inplace=True, axis=1)
from datetime import datetime as dt

train['datetime'] = pd.to_datetime(train['datetime'])

train['month'] = train['datetime'].dt.month

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train['year'] = train['datetime'].dt.year

train['year'] = train['year'].map({2011:0,2012:1})

train.head()
train.drop('datetime', axis=1, inplace=True)

train.head()
test['datetime'] = pd.to_datetime(test['datetime'])

test['month'] = test['datetime'].dt.month

test['day'] = test['datetime'].dt.day

test['hour'] = test['datetime'].dt.hour

test['year'] = test['datetime'].dt.year

test['year'] = test['year'].map({2011:0,2012:1})



test.head()
fig = plt.gcf()

fig.set_size_inches(10,8)

sns.heatmap(data = train.corr())
train2 = train.copy()
train2['temp_bins'] = np.floor(train2['temp'])//5

train2.head()

sns.factorplot(x='temp_bins', y='count', data=train2, kind='bar')
train.drop(['casual','registered'],axis=1,inplace=True)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train.drop('count',axis=1),train['count'],test_size=0.25,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]

model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']

rmsle=[]

d={}



for i in range(len(model_names)):

    clf = models[i]

    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)

    rmsle.append(np.sqrt(mean_squared_log_error(pred, y_test)))

d={'Modelling Algo':model_names,'RMSLE':rmsle}   

d
pd.DataFrame(d)
no_of_test=[500]

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf.fit(x_train,y_train)

pred=clf_rf.predict(x_test)

print((np.sqrt(mean_squared_log_error(pred,y_test))))
test.head()
pred2=clf_rf.predict(test.drop('datetime', axis=1))

d={'datetime':test['datetime'],'count':pred2}

ans=pd.DataFrame(d)

ans.to_csv('answer.csv',index=False)