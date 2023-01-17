# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train_V2.csv")
test=pd.read_csv("../input/test_V2.csv")
sample_submission=pd.read_csv("../input/sample_submission_V2.csv")
train.head()
train.columns
train.dtypes
print('Shape of train data=',train.shape)
print('Shape of test data=',test.shape)

#cheking for nulls in train data
train.isnull().sum().sort_values(ascending=False).head(3)
# one null in column winplacePerc
#cheking for nulls in test data
test.isnull().sum().sort_values(ascending=False).head(3)
#dropping the null values
train=train.dropna(how='any',axis=0)
#confirming the removal of nulls
train.isnull().sum().sort_values(ascending=False).head(3)
#Checking the distribution of target variable
plt.figure(figsize=(12,10))
sns.distplot(train.winPlacePerc)
#distribution of target varible is not normal
#let's Reduce the dimension of data then we will perform multivariate analysis
train.drop(columns=['Id','groupId','matchId'],inplace=True,axis=1)
test.drop(columns=['Id','groupId','matchId'],inplace=True,axis=1)
plt.figure(figsize=(16,14))
sns.heatmap(train.corr(),annot=True)
#finding correlation to pick most correlated features
corr=train.corr()['winPlacePerc']
corr[abs(corr)>0.1]
#let's pick these variable and make new train_corr data & test_corr
train_corr=train[['assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace','kills','killStreaks','longestKill','revives','rideDistance','swimDistance','walkDistance','weaponsAcquired','winPlacePerc']]
test_corr=test[['assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace','kills','killStreaks','longestKill','revives','rideDistance','swimDistance','walkDistance','weaponsAcquired']]
# lets See theshape of data now
print('The train data has {} datapoints an {} features now.'.format(train_corr.shape[0],train_corr.shape[1]))
print('The test data has {} datapoints an {} features now.'.format(test_corr.shape[0],test_corr.shape[1]))
from sklearn.linear_model import Lasso
ls=Lasso()
ls.fit(train_corr.drop(columns='winPlacePerc',axis=1),train['winPlacePerc'])
column_contribution=ls.coef_
column_contribution=pd.Series(column_contribution)
column_name=(train_corr.drop(columns='winPlacePerc',axis=1)).columns
column_name=pd.Series(column_name)
contributing_columns=pd.concat([column_name,column_contribution],axis=1)
contributing_columns
#Let's select thest features for univariate analysis
final_train=train_corr[['damageDealt','killPlace','rideDistance','walkDistance','winPlacePerc']]
final_test=test_corr[['damageDealt','killPlace','rideDistance','walkDistance']]
final_train.head()
# As you can see it will remove few of the best feature like boosts,heals,weaponsAcquired
# Since it removes multicollinear feature but we can't remove those feature
train_corr.head(4)
#let split and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_corr.drop(columns='winPlacePerc',axis=1),train_corr['winPlacePerc'],test_size=0.2)
from xgboost import XGBRegressor
xgb=XGBRegressor(max_depth=4,learning_rate=0.1,n_jobs=4)
#xgb.fit(x_train,y_train)

#checking accuracy
y_test_xg=xgb.predict(x_test)
from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(y_test,y_test_xg))
print(r2_score(y_test,y_test_xg))
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
x_train_ls,x_test_ls,y_train_ls,y_test_ls=train_test_split(final_train.drop(columns='winPlacePerc',axis=1),final_train['winPlacePerc'],test_size=0.2)
#xgb.fit(x_train_ls,y_train_ls)
#checking accuracy
y_test_xg_ls=xgb.predict(x_test_ls)
from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(y_test,y_test_xg_ls))
print(r2_score(y_test,y_test_xg_ls))
from sklearn.linear_model import LinearRegression
ls=LinearRegression()
#ls.fit(x_train_ls,y_train_ls)
y_test_li_ls=ls.predict(x_test_ls)
from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(y_test,y_test_li_ls))
print(r2_score(y_test,y_test_li_ls))
