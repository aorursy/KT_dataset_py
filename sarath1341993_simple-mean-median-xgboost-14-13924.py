# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import xgboost as xgb
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

df=pd.concat([train,test])
df.head()
df.isnull().sum()
df['date']=pd.to_datetime(df['date'])
df.dtypes
def cols_new(data_df):
    data_df['year'] = data_df['date'].dt.year
    data_df['quarter'] = data_df['date'].dt.quarter
    data_df['month'] = data_df['date'].dt.month
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    #data_df['weekday'] = data_df['date'].dt.weekday
    data_df['dayofweek'] = data_df['date'].dt.dayofweek
    return data_df
cols_new(df)
df.groupby(['item','store'])['sales'].median()
df.columns
 %%time
def mean_cols(data,cols):
    for i in cols:
        cols=[e for e in cols if e not in (i)]
        for j in cols :
            if i!=j :
                data['mean_'+i+'_'+j]=data.groupby([i,j])['sales'].transform('mean')
    return data
df.columns
%%time
mean_cols(df,['item','store','dayofweek','weekofyear','month','quarter'])
print(df.columns)
df.shape
def median_cols(data,cols):
    for i in cols:
        cols=[e for e in cols if e not in (i)]
        for j in cols :
            if i!=j :
                data['median_'+i+'_'+j]=data.groupby([i,j])['sales'].transform('median')
    return data
%%time
median_cols(df,['item','store','dayofweek','weekofyear','month','quarter'])
print(df.columns)
df.shape
df.head()
train = df.loc[~df.sales.isna()]
test = df.loc[df.sales.isna()]

print(train.shape,test.shape)
train.isnull().sum().sum()
X_train = train.drop(['date','sales','id'], axis=1)
y_train = train['sales'].values
X_test = test.drop(['id','date','sales'], axis=1)
X_train.isnull().sum().sum()
x_train, x_validate, y_train, y_validate = train_test_split(X_train, y_train, random_state=100, test_size=0.25)
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
xgbmodel = xgb.train(list(params.items()), xgbtrain, early_stopping_rounds=50,
                     evals=[(xgbtrain, 'train'), (xgbvalidate, 'validate')], 
                     num_boost_round=200, verbose_eval=50)
%%time
model = xgbmodel


predict=pd.DataFrame(model.predict(xgb.DMatrix(X_test),ntree_limit=model.best_ntree_limit),columns=['sales'])
ids=pd.read_csv("../input/test.csv",usecols=['id'])
predict=np.round(predict)
sub=ids.join(predict)
sub.head()
sub.to_csv('xgb_grpby_mean_median.csv',index=False)
