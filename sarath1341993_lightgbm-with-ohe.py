import pandas as pd
import numpy as np
import os
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
print("train data size:",train.shape)
print("test data size:",test.shape)

print("train:\n",train.head())
print("test:\n",test.head())
train['train/test']='train'
test['train/test']='test'
target=train.sales
train.drop('sales',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)
df=pd.concat([train,test])
df.head()
print(df.shape,"\n",df.dtypes)
df['date']=pd.to_datetime(df['date'])
df['dayofmonth'] = df.date.dt.day.astype(str)
df['dayofyear'] = df.date.dt.dayofyear.astype(str)
df['dayofweek'] = df.date.dt.dayofweek.astype(str)
df['month'] = df.date.dt.month.astype(str)
#df['year'] = df.date.dt.year.astype(str)
df['weekofyear'] = df.date.dt.weekofyear.astype(str)
#df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
#df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
df[360:370]
df.dtypes
trntst=df['train/test']
df.drop('train/test',axis=1,inplace=True)
df.head()
%%time
df=pd.get_dummies(df)
df.shape
df['train_or_test']=trntst
del trntst
df.shape
df.drop('date',axis=1,inplace=True)
x=df[df['train_or_test']=='train']
test=df[df['train_or_test']=='test']
x.drop('train_or_test',axis=1,inplace=True)
test.drop('train_or_test',axis=1,inplace=True)
print(x.shape,test.shape)
import lightgbm as lgb
lg=lgb.LGBMRegressor()
lg
%%time
lg.fit(x,target)
lg.score(x,target)
%%time
predict=pd.DataFrame(lg.predict(test),columns=['sales'])
predict.head()
ids=pd.read_csv("../input/test.csv",usecols=['id'])
sub=ids.join(predict)
sub.to_csv("lgbm_get_dummies.csv",index=False)