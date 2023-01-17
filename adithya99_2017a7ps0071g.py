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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)
train= pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

test= pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")
train_a=train.loc[:,['time', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5','a6']]

test_a=test.loc[:,['time', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5','a6']]

y_train=train.loc[:,'label'].values

final_id=test['id']
train.drop(columns=['id','time', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5','a6', 'label'], inplace=True)

test.drop(columns=['id','time', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5','a6'], inplace=True)
# train.drop(columns=['id', 'label'], inplace=True)

# test.drop(columns=['id'], inplace=True)
train1=train.copy()

test1=test.copy()
dropped=[]

for i in list(train1.columns):

    if(train[i].std()==0):

        dropped.append(i)

        
dropped
train1.drop(columns=dropped, inplace=True)

test1.drop(columns=dropped, inplace=True)
final_features=['b0',

 'b3',

 'b4',

 'b7',

 'b8',

 'b11',

 'b15',

 'b19',

 'b20',

 'b22',

 'b23',

 'b24',

 'b27',

 'b29',

 'b30',

 'b31',

 'b32',

 'b33',

 'b34',

 'b37',

 'b38',

 'b41',

 'b42',

 'b48',

 'b50',

 'b51',

 'b52',

 'b59',

 'b62',

 'b65',

 'b66',

 'b69',

 'b76',

 'b77',

 'b78',

 'b79',

 'b80',

 'b82',

 'b84',

 'b85',

 'b86',

 'b87',

 'b90',

 'b93']
train1=train[final_features].copy()

test1=test[final_features].copy()
train1.columns
test1.columns
train1=pd.concat([train1, train_a], axis=1)

test1=pd.concat([test1,test_a], axis=1)
print(train1.shape)

print(test1.shape)

X_train=train1.iloc[:,:].values

X_test=test1.iloc[:,:].values
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
# from sklearn.preprocessing import MinMaxScaler

# scaler_x = MinMaxScaler()

# scaler_y = MinMaxScaler()

from sklearn.preprocessing import RobustScaler

scaler_x = RobustScaler(quantile_range = (0.2,0.8))

X_train=scaler_x.fit_transform(X_train)

X_test=scaler_x.transform(X_test)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

# regressor=HistGradientBoostingRegressor(l2_regularization=1.25, max_iter=1350, max_depth=35,warm_start=True, scoring='neg_root_mean_squared_error', learning_rate=0.127, random_state=0)

regressor = RandomForestRegressor(n_estimators =500, n_jobs=-1, random_state=0 )



regressor.fit(X_train, y_train.reshape(-1))

y_pred = regressor.predict(X_test)

y_pred_train=regressor.predict(X_train)

mae_lr = mean_squared_error(y_pred_train,y_train, squared=False)

mse = mean_absolute_error(y_pred_train,y_train)



print("Mean Absolute Error of Linear Regression: {}".format(mse))



print("Mean Absolute Error of Linear Regression: {}".format(mae_lr))
y_pred.shape
#y_pred_lr=scaler_value.inverse_transform(y_pred_lr.reshape(-1,1))

y_pred = pd.Series(y_pred)  

  

frame = { 'id': final_id, 'label': y_pred } 

  

result = pd.DataFrame(frame)
result.head()
filename = 'submission_5.csv'



result.to_csv(filename,index=False)
final_id