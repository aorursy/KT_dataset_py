%%time
import pandas as pd

import  lightgbm as lgb
%%time
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
print(df_train.head(3))
print(df_test.head(3))
# Define column date as datatype date and define new date features
for dataset in combine:
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['year'] = dataset.date.dt.year
    dataset['month'] = dataset.date.dt.month
    dataset['day'] = dataset.date.dt.day
    dataset['dayofyear'] = dataset.date.dt.dayofyear
    dataset['dayofweek'] = dataset.date.dt.dayofweek
    dataset['weekofyear'] = dataset.date.dt.weekofyear
dataset.drop('date', axis=1, inplace=True)
df_train.head()
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
df_test.columns
df_train.shape
df_test.shape
df_train.head(2)
df_test.head(2)
df_train.isnull().sum()
df_test.isnull().sum()
df_train.dtypes
df_test.dtypes
#setting parameters for lightgbm
param = {'num_leaves':150, 'max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']

y=pd.DataFrame()
y=df_train['sales']

df_train=df_train.drop(['sales'],axis=1)
x=df_train

%%time
train_data = lgb.Dataset(x,y)
model =lgb.train(param,train_data,)
%%time
output=model.predict(df_test)
result=pd.DataFrame(output)
result
%%time
test=pd.read_csv('../input/test.csv',usecols=['id'])
fin=pd.DataFrame(test)
fin['sales']=result
fin.to_csv('Sales_Lgm.csv',index=False)
 