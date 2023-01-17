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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

train.describe()
train.head()
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import missingno as msno

train.shape
var1={"store_and_fwd_flag":{"Y":1,"N":0}}
train.replace(var1,inplace=True)
test.replace(var1,inplace=True)
test["store_and_fwd_flag"][:10]
train.isnull().sum()

num_feat=train.select_dtypes(include=[np.number])
num_feat.columns

correlation=num_feat.corr()
print(correlation['trip_duration'].sort_values(ascending=False))
f, ax=plt.subplots(figsize=(14,12))
sns.heatmap(correlation,square=True,vmax=0.8)


train.drop("pickup_datetime",axis=1,inplace=True)
test.drop("pickup_datetime",axis=1,inplace=True)
train.head()
train.drop("dropoff_longitude",axis=1,inplace=True)
test.drop("dropoff_longitude",axis=1,inplace=True)
train.head()
train.drop("dropoff_datetime",axis=1,inplace=True)
test.drop("dropoff_datetime",axis=1,inplace=True)
train.head()
id1=test.id
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)
train.head()
train.head()
Y_train=train.trip_duration
X_train=train.drop("trip_duration",axis=1)
Y_test=test.trip_duration
X_test=test.drop("trip_duration",axis=1)

from xgboost import XGBRegressor
my_model=XGBRegressor(n_estimators=300,learning_rate=0.2)
my_model.fit(X_train,Y_train,verbose=False)

predict=my_model.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test,predict))

my_submission = pd.DataFrame({'Id': id1, 'Trip_Duration': predict})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
