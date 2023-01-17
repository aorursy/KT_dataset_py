# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')
test_df=test.copy()
train_df=train.copy()
train.head()
train_df["hour"] = [t.hour for t in pd.DatetimeIndex(train_df.datetime)]
train_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_df.datetime)]
train_df["month"] = [t.month for t in pd.DatetimeIndex(train_df.datetime)]
train_df['year'] = [t.year for t in pd.DatetimeIndex(train_df.datetime)]
train_df['year'] = train_df['year'].map({2011:0, 2012:1})

test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]
test_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_df.datetime)]
test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]
test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]
test_df['year'] = test_df['year'].map({2011:0, 2012:1})
train_df.head()
train_df.drop('datetime',axis=1,inplace=True)
train_df.head()
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots(2,2)
sns.barplot(train_df['season'],train_df['count'],ax=ax[0,0]);
sns.barplot(train_df['holiday'],train_df['count'],ax=ax[0,1]);
sns.barplot(train_df['workingday'],train_df['count'],ax=ax[1,0]);
sns.barplot(train_df['weather'],train_df['count'],ax=ax[1,1]);
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots(2,2)
sns.distplot(train_df['temp'],ax=ax[0,0]);
sns.distplot(train_df['atemp'],ax=ax[0,1]);
sns.distplot(train_df['humidity'],ax=ax[1,0]);
sns.distplot(train_df['windspeed'],ax=ax[1,1]);
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(train_df.corr(),annot=True,linewidths=0.5);
train_df.drop(['casual','registered'],axis=1,inplace=True)
sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(x=train_df['month'],y=train_df['count']);
season=pd.get_dummies(train_df['season'],prefix='season')
train_df=pd.concat([train_df,season],axis=1)
train_df.head()
season=pd.get_dummies(test_df['season'],prefix='season')
test_df=pd.concat([test_df,season],axis=1)
test_df.head()
weather=pd.get_dummies(train_df['weather'],prefix='weather')
train_df=pd.concat([train_df,weather],axis=1)
train_df.head()
weather=pd.get_dummies(test_df['weather'],prefix='weather')
test_df=pd.concat([test_df,weather],axis=1)
test_df.head()
train_df.drop(['season','weather'],inplace=True,axis=1)
train_df.head()
test_df.drop(['season','weather'],inplace=True,axis=1)
test_df.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train,x_test,y_train,y_test=train_test_split(train_df.drop('count',axis=1),train_df['count'],test_size=0.25,random_state=42)
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
no_of_test=[500]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')
rf.fit(x_train,y_train)
print(x_test)
pred = rf.predict(x_test)
print(len(pred))
print((np.sqrt(mean_squared_log_error(pred,y_test))))
print(list(pred))
pred=rf.predict(test_df.drop('datetime',axis=1))
d={'datetime':test['datetime'],'count':pred}
ans=pd.DataFrame(d)
ans.to_csv('bikeSharing.csv',index=False) 