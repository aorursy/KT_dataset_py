# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

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
test_def=test.copy()
train.head()
train['year'] = [t.year for t in pd.DatetimeIndex(train.datetime)]
train['month'] = [t.month for t in pd.DatetimeIndex(train.datetime)]
train['day'] = [t.day for t in pd.DatetimeIndex(train.datetime)]
train['hour'] = [t.hour for t in pd.DatetimeIndex(train.datetime)]

test['year'] = [t.year for t in pd.DatetimeIndex(test.datetime)]
test['month'] = [t.month for t in pd.DatetimeIndex(test.datetime)]
test['day'] = [t.day for t in pd.DatetimeIndex(test.datetime)]
test['hour'] = [t.hour for t in pd.DatetimeIndex(test.datetime)]
train.head()
train.drop('datetime',axis=1,inplace=True)
test.drop('datetime',axis=1,inplace=True)
train.head()
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots(2,2)
sns.barplot(train['season'],train['count'],ax=ax[0,0]);
sns.barplot(train['holiday'],train['count'],ax=ax[0,1]);
sns.barplot(train['workingday'],train['count'],ax=ax[1,0]);
sns.barplot(train['weather'],train['count'],ax=ax[1,1]);
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig, ax = plt.subplots(2,2)
sns.distplot(train['temp'],ax=ax[0,0]);
sns.distplot(train['atemp'],ax=ax[0,1]);
sns.distplot(train['humidity'],ax=ax[1,0]);
sns.distplot(train['windspeed'],ax=ax[1,1]);
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(train.corr(),annot=True,linewidths=0.5);
train.drop(['casual','registered'],axis=1,inplace=True)
sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(x=train['month'],y=train['count']);
season = pd.get_dummies(train['season'],prefix='season')
train = pd.concat([train,season],axis=1)
train.drop('season',axis=1,inplace=True)
train.head()
weather = pd.get_dummies(train['weather'],prefix='weather')

train = pd.concat([train,weather],axis=1)

train.drop('weather',axis=1,inplace=True)
train.head()
train.columns.to_series().groupby(train.dtypes).groups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = train.drop('count',axis=1)
y = train['count']
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# print(y_test)
# print(len(y_test))
# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor()
# model_rf = rf.fit(X_train,y_train)
# # print(y_train)
# # print(len(y_train))
# y_pred_rf = model_rf.predict(X_test)
# np.sqrt(mean_squared_log_error(y_test,y_pred_rf))
# # print(y_pred_rf)
# # print(len(y_pred_rf))
# # print(X_test)
# # print(len(X_test))
# # output = pd.read_csv('../input/bike-sharing-demand/sampleSubmission.csv', header = 0, sep = ',')
# # datetimecol = output['datetime']
# # # print(datetimecol)
# # submission = pd.DataFrame({
# #         "datetime": datetimecol,
# #         "count": [max(0, x) for x in np.exp(y_pred_rf)]
# #     })
# # print(submission)
# # submission.to_csv('../output/sampleSubmissionRandomForest.csv', index=False)

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
no_of_test=[500]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')
rf.fit(X_train,y_train)
print(X_test)
pred = rf.predict(X_test)
print(len(pred))
print((np.sqrt(mean_squared_log_error(pred,y_test))))
rf.best_params_
# print(test_def)
# test_df = test.drop('datetime',axis=1)
print(test)
pred=rf.predict(test)
d={'datetime':test_def['datetime'],'count':pred}
ans=pd.DataFrame(d)
ans.to_csv('sampleSubmissionRandomForest.csv',index=False)