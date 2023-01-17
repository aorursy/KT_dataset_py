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
import matplotlib.pyplot as plt

import seaborn as sns
building=pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

#weather_train=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

train=pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
test=pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

#weather_test=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
train=pd.merge(building,train,on='building_id',how='right')

test=pd.merge(building,test,on='building_id',how='right')
#d1.info()
train['timestamp']=pd.to_datetime(train.timestamp)

test['timestamp']=pd.to_datetime(test.timestamp)
#d2.info()
#d1.shape
#d2['timestamp']=d2.timestamp.dt.date

#d2=d2.groupby('timestamp').mean().reset_index()

#t2['timestamp']=t2.timestamp.dt.date

#t2=t2.groupby('timestamp').mean().reset_index()
#d2.shape
print(train.shape)

print(test.shape)
print(train.isnull().sum())

print(test.isnull().sum())
features=train[['square_feet','meter']]

y=train.meter_reading
test_X=test[['square_feet','meter']]
from sklearn.preprocessing import StandardScaler

x=StandardScaler().fit_transform(features)

test_x=StandardScaler().fit_transform(test_X)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()

rf.fit(x_train,y_train)

y_hat=rf.predict(x_test)
from sklearn.metrics import mean_squared_log_error as msle

(msle(y_test,y_hat))
test_y=rf.predict(test_x)
2.6434394121439375**0.5

sample=pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
sample.head()
test=pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
test['meter_reading']=test_y
submission=test[['row_id','meter_reading']]
submission.to_csv('energy.csv',index=False)
sub=pd.read_csv('energy.csv')
sub.head()