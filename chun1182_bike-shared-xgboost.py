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
!pip install category_encoders
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

ans = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

print("Train shape : ", train.shape)

print("Test shape : ", test.shape)

print("ans shape : ", ans.shape)



train
test
ans.head()
train.dtypes
train['datetime'] = pd.to_datetime(train['datetime'])

test['datetime'] = pd.to_datetime(test['datetime'])
train['year'] = train['datetime'].dt.year-2011

test['year'] = test['datetime'].dt.year-2011

train['weekday'] = train['datetime'].dt.weekday

test['weekday'] = test['datetime'].dt.weekday

train['weekday2'] = (train['datetime'].dt.weekday+3)%7

test['weekday2'] = (test['datetime'].dt.weekday+3)%7

train['hour'] = train['datetime'].dt.hour

test['hour'] = test['datetime'].dt.hour

train['hour2'] = (train['datetime'].dt.hour+12)%24

test['hour2'] = (test['datetime'].dt.hour+12)%24

train['month'] = train['datetime'].dt.month

test['month'] = test['datetime'].dt.month

train['month2'] = (train['datetime'].dt.month+6)%12

test['month2'] = (test['datetime'].dt.month+6)%12

train['day'] = train['datetime'].dt.day

test['day'] = test['datetime'].dt.day

train['tasu'] = train['year']*365 + train['month']*30 + train['day']

test['tasu'] = train['year']*365 + test['month']*30 + test['day']
train.describe()
train.groupby('month2').mean()
train['count'].hist(bins=20)
train['count'].describe()
train.columns
from sklearn.model_selection import train_test_split

import category_encoders as ce



y_train = np.log1p(train["casual"])

y_train2 = np.log1p(train["registered"])

y_train3 = np.log1p(train["count"])

y_mean = np.mean(y_train)

list_cols = ['season', 'weather']

use_columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',

               'year', 'weekday', 'weekday2', 'hour', 'hour2', 'month', 'month2']



ce_ohe = ce.OneHotEncoder(cols=list_cols)

train_onehot = ce_ohe.fit_transform(train[use_columns])
test_onehot = ce_ohe.transform(test[use_columns])

test_onehot.describe()
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score



gbk3 = GradientBoostingRegressor()

gbk3.fit(train_onehot, y_train3)
from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestRegressor()

rfc.fit(train_onehot, y_train3)

#y_pred = rfc.predict(x_val)

#acc_rfc = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_rfc)
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(max_iter=100, hidden_layer_sizes=(100,100), 

                    activation='relu',  learning_rate_init=0.01)

mlp.fit(train_onehot, y_train3)

#y_pred = mlp.predict(x_val)

#acc_rfc = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_rfc)
preds =  (np.expm1(gbk3.predict(test_onehot))+np.expm1(rfc.predict(test_onehot))

          +np.expm1(mlp.predict(test_onehot)))/3

preds = np.where(preds < 0 , 0, preds)
ans["count"] = preds

ans.to_csv("bike_sharing.csv", index=False)
ans['count'].hist(bins=20)
ans['count'].describe()
fti = gbk.feature_importances_  

print('Feature Importances:')

for i,feat in enumerate(train_onehot.columns):

    print('\t{0:10s} : {1:>12.4f}'.format(feat, fti[i]))