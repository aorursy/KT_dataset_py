import numpy as np # linear algebra
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/train.csv')
df_test= pd.read_csv('../input/test.csv')

def convert_dates(x):
    x['date']=pd.to_datetime(x['date'])
    x['month']=x['date'].dt.month
    x['year']=x['date'].dt.year
    x['dayofweek']=x['date'].dt.dayofweek
    x.pop('date')
    return x

df = convert_dates(df_train)
df_test = convert_dates(df_test)

def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','dayofweek'])['sales'].transform('mean')
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean')
    return x
df_train = add_avg(df_train)
daily_avg = df_train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg = df_train.groupby(['item','store','month'])['sales'].mean().reset_index()

def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x
df_test=merge(df_test, daily_avg,['item','store','dayofweek'],'daily_avg')
df_test=merge(df_test, monthly_avg,['item','store','month'],'monthly_avg')

x_train,x_test,y_train,y_test = train_test_split(df_train.drop('sales',axis=1),df_train.pop('sales'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                    ,dtrain=matrix_train,num_boost_round=500, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
df_test=df_test.drop(['id'],axis=1)
%%time
submission =  pd.read_csv('../input/test.csv',usecols=['id'])
y_pred = model.predict(xgb.DMatrix(df_test), ntree_limit = model.best_ntree_limit)

submission['sales']= y_pred

submission.to_csv('submission1.csv',index=False)