%%time
import pandas as pd

import  lightgbm as lgb
%%time

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print(df_train.head(3))
print(df_test.head(3))
# Define column date as datatype date and define new date features
def add_new_features(x):
    x['date'] = pd.to_datetime(x['date'])
    x['year'] = x.date.dt.year
    x['month'] = x.date.dt.month
    x['dayofweek'] = x.date.dt.dayofweek
    return x
df_train = add_new_features(df_train)
df_test= add_new_features(df_test)

df_train
df_train['daily_avg']=df_train.groupby(['item','store','dayofweek'])['sales'].transform('mean')
df_train['monthly_avg']=df_train.groupby(['item','store','month'])['sales'].transform('mean')
daily_avg=df_train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg=df_train.groupby(['item','store','month'])['sales'].mean().reset_index()

monthly_avg
def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    
    x=x.rename(columns={'sales':col_name})
    return x

df_test=merge(df_test, daily_avg,['item','store','dayofweek'],'daily_avg')
df_test=merge(df_test, monthly_avg,['item','store','month'],'monthly_avg')

print(df_test.columns)
print(df_train.columns)

df_test=df_test.drop(['id'],axis=1)
df_train=df_train.drop(['date'],axis=1)
df_test=df_test.drop(['date'],axis=1)
df_test.columns
df_train.shape
df_test.shape
df_train.head(2)
df_test.head(2)
df_train.isnull().sum()
df_test.isnull().sum()
df_train.dtypes
df_test.dtypes

y=pd.DataFrame()
y=df_train['sales']

df_test.head(2)
df_train.head()
import xgboost as xgb
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df_train.drop('sales',axis=1),df_train.pop('sales'),random_state=123,test_size=0.2)
def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                    ,dtrain=matrix_train,num_boost_round=500, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)

df_test
df_test1 = pd.read_csv('../input/test.csv')
submission = pd.DataFrame(df_test1.pop('id'))
y_pred = model.predict(xgb.DMatrix(df_test), ntree_limit = model.best_ntree_limit)

submission['sales']= y_pred

submission.to_csv('submission.csv',index=False)
submission.tail()

x_test.describe()
