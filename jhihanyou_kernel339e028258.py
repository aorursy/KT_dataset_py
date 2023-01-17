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
train = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv") 

test = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv") 
import numpy as np

import matplotlib.pyplot as plt

import datetime

from sklearn.model_selection import train_test_split, TimeSeriesSplit

import xgboost as xgb

from sklearn.metrics import mean_absolute_error
test.columns
train.columns
print(train.shape,test.shape)
test.tail()
print('store'	, ' values: ', train['store'].unique())

print('item'	, ' values: ', train['item'].unique())

print('store'	, ' values: ', test['store'].unique())

print('item'	, ' values: ', test['item'].unique())
# plot the data (item=1,store=1) in different time scale

plt.subplot(3, 1, 1)

plt.plot(train.sales[:365*4])

plt.subplot(3, 1, 2)

plt.plot(train.sales[:365])



plt.subplot(3, 1, 3)

plt.plot(train.sales[:31])

plt.show()
# plot the moving average of the data (item=1,store=1) in different time scale

rolling_mean = train.sales.rolling(window=7).mean()

plt.subplot(3, 1, 1)

plt.plot(rolling_mean[:365*4])

plt.subplot(3, 1, 2)

plt.plot(rolling_mean[:365])



plt.subplot(3, 1, 3)

plt.plot(rolling_mean[:31])

plt.show()
data_combine = pd.concat([train,test])



print("size of data_combine",data_combine.shape)
data_combine['date'] = pd.to_datetime(data_combine['date'],infer_datetime_format=True)

data_combine['month'] = data_combine['date'].dt.month

data_combine['weekday'] = data_combine['date'].dt.dayofweek

data_combine['year'] = data_combine['date'].dt.year

# df['date'].dt.

data_combine['week_of_year']  = data_combine.date.dt.weekofyear



data_combine['date_order'] = (data_combine['date'] - datetime.datetime(2013, 1, 1)).dt.days
data_combine.head(40)
# To calculate the moving averages of 7 days in order to smooth the noise signal

data_combine['sale_moving_average_7days']=data_combine.groupby(["item","store"])['sales'].transform(lambda x: x.rolling(window=7,min_periods=1).mean())

data_combine['sale_moving_average_7days_shifted-90']=data_combine.groupby(["item","store"])['sale_moving_average_7days'].transform(lambda x:x.shift(90))



# To get the sale price 90 days ago as one of the feature because we have future 90 days to be predicted...

data_combine['price_shifted-90'] = data_combine.groupby(["item","store"])['sales'].transform(lambda x:x.shift(90))
data_combine.head(n=40)
col = [i for i in data_combine.columns if i not in ['date','id','sale_moving_average_7days']]
train_new = data_combine.loc[~data_combine.sales.isna()]

print("new train",train_new.shape)

test_new = data_combine.loc[data_combine.sales.isna()]

print("new test",test_new.shape)
train_new = (train_new[col]).dropna()

print(train_new.shape)
y_target = train_new.sales

col = [i for i in data_combine.columns if i not in ['date','id','sales','sale_moving_average_7days']]
from sklearn.model_selection import train_test_split, TimeSeriesSplit

X_train, X_test, y_train, y_test = train_test_split( train_new[col] ,y_target, test_size=0.15)
def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
model_sets=[]

for max_depth in range(4,14,3):

  xgb_model = xgb.XGBRegressor(max_depth=max_depth ,min_child_weight=1)

  xgb_model.fit(X_train,y_train.values,eval_metric=smape)

  model_sets.append(xgb_model)

  

  y_train_pred_xgb=xgb_model.predict(X_train)

  y_test_pred_xgb=xgb_model.predict(X_test)

  print('smape error: max_depth=', max_depth ,',train:' , smape(y_train.values,y_train_pred_xgb),'test:',smape(y_test.values,y_test_pred_xgb))


#xgb_model = xgb.XGBRegressor(max_depth=8,min_child_weight=1)

model = model_sets[2]

model.fit(train_new[col],y_target,eval_metric=smape)



y_train_pred_xgb=model.predict(train_new[col])

print('smape error: ' , smape(y_target,y_train_pred_xgb))

#y_test_pred_xgb=model.predict(X_test)

#print('smape error: max_depth=', max_depth ,',train:' , smape(y_train.values,y_train_pred_xgb),'test:',smape(y_test.values,y_test_pred_xgb))

#print('MSE train:' , mean_absolute_error(np.log1p(y_train),np.log1p(y_train_pred_xgb)),'test:',mean_absolute_error(np.log1p(y_test),np.log1p(y_test_pred_xgb)))
# Choose the model which provides less smape

y_submission=np.rint(model.predict(test_new[col]))

#y_submission=xgb_model.predict(test_new[col])
final = pd.DataFrame(list(zip(test_new['id'], y_submission)), 

               columns =['id', 'sales']) 
final.head()
final.id = final.id.astype(int)

final.sales = final.sales.astype(int)
final.head(n=8)
final.to_csv("submission.csv",sep=',', index=False)