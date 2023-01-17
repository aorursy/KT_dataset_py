%matplotlib inline

import pandas as pd

from datetime import datetime

import pandas as pd

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import mean_squared_error

from math import radians, cos, sin, asin, sqrt

import seaborn as sns

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



#Load Data 

train = pd.read_csv("../input/taxiny/train.csv")

test = pd.read_csv('../input/taxiny/test.csv')

sample = pd.read_csv('../input/taxiny/sample_submission.csv')

print("fini")



pd.set_option('display.float_format', lambda x: '%.5f' % x)

print(train)
### On élimine les données absurdes "trip duration"

import numpy as np

import pandas as pd



m = np.mean(train['trip_duration'])

s = np.std(train['trip_duration'])



#print("La moyenne est : ", m)

#print("L'écart-type est : ", s)



train = train[train['trip_duration'] <= m + 2*s]

train = train[train['trip_duration'] >= m - 2*s]





#Clean-up

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime) #Not in Test



train = train.drop(train[train.pickup_latitude < 40.1].index)

train = train.drop(train[train.pickup_latitude > 41.4].index)

train = train.drop(train[train.pickup_longitude >-73].index)

train = train.drop(train[train.pickup_longitude < -86].index)



train = train.drop(train[train.dropoff_latitude < 39].index)

train = train.drop(train[train.dropoff_latitude > 42].index)

train = train.drop(train[train.dropoff_longitude >-30].index)

train = train.drop(train[train.dropoff_longitude < -75].index)



train = train.drop(train[train.trip_duration < 45].index)

train = train.drop(train[train.trip_duration > 30000].index)
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from datetime import datetime





#Extracting Month

train['Month'] = train['pickup_datetime'].dt.month

test['Month'] = test['pickup_datetime'].dt.month

train.groupby('Month').size(), test.groupby('Month').size()





#Extracting Day

train['DayofMonth'] = train['pickup_datetime'].dt.day

test['DayofMonth'] = test['pickup_datetime'].dt.day

train.groupby('DayofMonth').size(), test.groupby('DayofMonth').size()





#Hours

train['Hour'] = train['pickup_datetime'].dt.hour

test['Hour'] = test['pickup_datetime'].dt.hour

print(train.groupby('Hour').size(), test.groupby('Hour').size())





#Weeks

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek

test['dayofweek'] = test['pickup_datetime'].dt.dayofweek

train.groupby('dayofweek').size(), test.groupby('dayofweek').size()





month_train = pd.get_dummies(train['Month'], prefix='m', prefix_sep='_')

month_test = pd.get_dummies(test['Month'], prefix='m', prefix_sep='_')

daym_train = pd.get_dummies(train['DayofMonth'], prefix='daym', prefix_sep='_')

daym_test = pd.get_dummies(test['DayofMonth'], prefix='daym', prefix_sep='_')

hour_train = pd.get_dummies(train['Hour'], prefix='h', prefix_sep='_')

hour_test = pd.get_dummies(test['Hour'], prefix='h', prefix_sep='_')

dayw_train = pd.get_dummies(train['dayofweek'], prefix='dayw', prefix_sep='_')

dayw_test = pd.get_dummies(test['dayofweek'], prefix='dayw', prefix_sep='_')



print(train)

train.to_csv('train_ohe')
#Data analyse 

from pandas.plotting import register_matplotlib_converters

import matplotlib.pyplot as plt

import matplotlib 

import seaborn as sns ##Permet de faire des graphes plus simplement qu'avec matplotlib



plt.hist(train['trip_duration'].values, bins=100)

plt.xlabel('trip_duration (seconde)')

plt.ylabel('number of train records')

plt.show()
sns.countplot(train['trip_duration'])
sns.barplot(x='dayofweek', y='trip_duration', data=train)
sns.barplot(x='Hour', y='trip_duration', data=train)
sns.barplot(x='DayofMonth', y='trip_duration', data=train)
sns.barplot(x='Month', y='trip_duration', data=train)
sns.barplot(x='Month', y='trip_duration', data=train)
plt.hist(train['Hour'].values, bins=100)

plt.xlabel('Hour') #0=00h, ... , 23=23h

plt.ylabel('number of train records')

plt.show()
plt.hist(train['dayofweek'].values, bins=100)

plt.xlabel('dayofweek')

plt.ylabel('number of train records')

plt.show()
plt.hist(train['DayofMonth'].values, bins=100)

plt.xlabel('Day of Month')

plt.ylabel('number of train records')

plt.show()
sns.countplot(train['DayofMonth'])
plt.hist(train['Month'].values, bins=100)

plt.xlabel('Month') # 1=janvier, 2=février, ... , 6=juin

plt.ylabel('number of train records')

plt.show()
plt.hist(train['dayofweek'].values, bins=100)

plt.xlabel('dayofweek')

plt.ylabel('trip_duration')

plt.show()
Train_master = pd.concat([month_train, daym_train, hour_train, dayw_train], axis=1)

Test_master = pd.concat([month_test, daym_test, hour_test, dayw_test], axis=1)



print("Train :", Train_master), print("Test:", Test_master)

Test_master.head(), Train_master.head()

Train, Test = train_test_split(Train_master[0:100000])



Train_master.to_csv('Train_ouput')

Test_master.to_csv('Test_ouput')