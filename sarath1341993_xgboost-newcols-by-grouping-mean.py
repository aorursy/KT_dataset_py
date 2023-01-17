import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.dtypes
train_df.head()
test_df.tail()
train_df['trsin_or_test'], test_df['trsin_or_test'] = 'train', 'test'
data_df = pd.concat([train_df, test_df])
data_df.head()
data_df['date']=pd.to_datetime(data_df['date'])
data_df.dtypes
data_df.info()
data_df['year'] = data_df['date'].dt.year
data_df['quarter'] = data_df['date'].dt.quarter
data_df['month'] = data_df['date'].dt.month
data_df['weekofyear'] = data_df['date'].dt.weekofyear
data_df['weekday'] = data_df['date'].dt.weekday
data_df['dayofweek'] = data_df['date'].dt.dayofweek
data_df.head()
data_df.groupby(['quarter', 'item'])['sales'].mean()
data_df['item_quarter_mean'] = data_df.groupby(['quarter', 'item'])['sales'].transform('mean')
data_df.head()
data_df['store_quarter_mean'] = data_df.groupby(['quarter', 'store'])['sales'].transform('mean')
data_df['store_item_quarter_mean'] = data_df.groupby(['quarter', 'item', 'store'])['sales'].transform('mean')
data_df['item_month_mean'] = data_df.groupby(['month', 'item'])['sales'].transform('mean')
data_df['store_month_mean'] = data_df.groupby(['month', 'store'])['sales'].transform('mean')
data_df['store_item_month_mean'] = data_df.groupby(['month', 'item', 'store'])['sales'].transform('mean')
data_df['item_weekofyear_mean'] = data_df.groupby(['weekofyear', 'item'])['sales'].transform('mean')
data_df['store_weekofyear_mean'] = data_df.groupby(['weekofyear', 'store'])['sales'].transform('mean')
data_df['store_item_weekofyear_mean'] = data_df.groupby(['weekofyear', 'item', 'store'])['sales'].transform('mean')

data_df['itemweekday_mean'] = data_df.groupby(['weekday', 'item'])['sales'].transform('mean')
data_df['storeweekday_mean'] = data_df.groupby(['weekday', 'store'])['sales'].transform('mean')
data_df['storeitemweekday_mean'] = data_df.groupby(['weekday', 'item', 'store'])['sales'].transform('mean')
data_df.head()
data_df.tail(n=3)
data_df.isnull().sum().sum()
data_df.info()
data_df.head()
data_df.shape
data_df.columns
data_df.drop(['date','id','sales'],axis=1,inplace=True)
data_df.info()
x= data_df[data_df['trsin_or_test'] == 'train']#.dropna().drop(['id', 'sales', 'trsin_or_test', 'date'], axis=1)
test = data_df[data_df['trsin_or_test'] == 'train']#.dropna()['sales']
x.head()
test.head()
x.drop(['trsin_or_test'],axis=1,inplace=True)
test.drop(['trsin_or_test'],axis=1,inplace=True)
y=pd.read_csv('../input/train.csv',usecols=['sales'])
y=y.sales
y.shape
y.head()
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=0, test_size=0.25)
print(x_train.shape, x_validate.shape, y_train.shape, y_validate.shape)
%%time
params = {
    'colsample_bytree': 0.8,
    'eta': 0.1,
    'eval_metric': 'mae',
    'lambda': 1,
    'max_depth': 6,
    'objective': 'reg:linear',
    'seed': 0,
    'silent': 1,
    'subsample': 0.8,
}
xgbtrain = xgb.DMatrix(x_train, label=y_train)
xgbvalidate = xgb.DMatrix(x_validate, label=y_validate)
xgbmodel = xgb.train(list(params.items()), xgbtrain, early_stopping_rounds=50, evals=[(xgbtrain, 'train'), (xgbvalidate, 'validate')], num_boost_round=200, verbose_eval=50)
%%time
model = xgbmodel


predict=pd.DataFrame(model.predict(xgb.DMatrix(test),ntree_limit=model.best_ntree_limit),columns=['sales'])

ids=pd.read_csv("../input/test.csv",usecols=['id'])
sub=ids.join(predict)
sub.head()
sub.to_csv('xgb_grpby_mean.csv',index=False)