import numpy as np # linear algebra
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/train.csv')
df_test= pd.read_csv('../input/test.csv')

def convert_dates(x):
    x['date']=pd.to_datetime(x['date'])
    x['month']=x['date'].dt.month
    x['year']=x['date'].dt.year
    x['dayofweek']=x['date'].dt.dayofweek
    return x

df_train = convert_dates(df_train)
df_test = convert_dates(df_test )

def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','dayofweek'])['sales'].transform('mean')
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean')
    return x
df_train = add_avg(df_train)
daily_avg = df_train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg = df_train.groupby(['item','store','month'])['sales'].mean().reset_index()

df_test=df_test.drop(['date'],axis=1)
df_train=df_train.drop(['date'],axis=1)
def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x
df_test=merge(df_test, daily_avg,['item','store','dayofweek'],'daily_avg')
df_test=merge(df_test, monthly_avg,['item','store','month'],'monthly_avg')

x_train,x_test,y_train,y_test = train_test_split(df_train.drop('sales',axis=1),df_train.pop('sales'),random_state=123,test_size=0.2)

def run_lgb(x_train,y_train,x_test,y_test):
    params = {
        "objective" : "regression",
        "metric" : "mape",
        "boosting_type":"gbdt",
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.6,
        "seed": 2018,
        "num_leaves":7,
        "max_depth":6,
    }
    
    lgtrain = lgb.Dataset(x_train, label=y_train)
    lgval = lgb.Dataset(x_test, label=y_test)
    evals_result = {}
    model = lgb.train(params, lgtrain, 10000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=50, 
                      verbose_eval=100, 
                      evals_result=evals_result)
    return model
model = run_lgb(x_train,y_train,x_test,y_test)
y_pred = model.predict(df_test)
submission = pd.read_csv('../input/test.csv',usecols=['id'])
y_pred = model.predict(df_test,num_iteration=model.best_iteration)

submission['sales']= y_pred

submission.to_csv('submission.csv',index=False)
df_test.columns
df_train.columns

x_test.describe()

