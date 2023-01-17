import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
from datetime import datetime as dt
import os
print(os.listdir("../input"))
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test_id = test['id']
train
for col in train.columns:
    print(train[col].value_counts())
m=np.mean(train['trip_duration'])
s=np.std(train['trip_duration'])
train=train[m-2*s<=train['trip_duration']]
train=train[train['trip_duration']<=m+2*s]
train['log_trip_duration']=np.log(train['trip_duration'].values+1)
sn.distplot(train['log_trip_duration'],bins=100)
vendor=train.groupby('vendor_id')['trip_duration'].mean()
vendor_plot=sn.barplot(vendor.index,vendor.values)
vendor_plot.set(ylim=(800,850))
df=pd.DataFrame(train.groupby('passenger_count')['trip_duration'].mean()).reset_index()
df
sn.barplot(x=df['passenger_count'],y=df['trip_duration'])
train[train['passenger_count']==0]
train['traveled_distance']=np.sqrt((train['pickup_latitude']-train['dropoff_latitude'])**2 + (train['pickup_longitude']-train['dropoff_longitude'])**2)
test['traveled_distance']=np.sqrt((test['pickup_latitude']-test['dropoff_latitude'])**2 + (test['pickup_longitude']-test['dropoff_longitude'])**2)
dist=sn.lmplot(x='traveled_distance',y='trip_duration',data=train,fit_reg=False)
dist.set(xlim=(0,1))
dist.set(ylim=(0,200000))
dist
store=train.groupby('store_and_fwd_flag')['trip_duration'].mean()
sn.barplot(x=store.index,y=store.values)
fig, [ax1,ax2]=plt.subplots(ncols=2)
box1=sn.boxplot(x=train[train['store_and_fwd_flag']=='N']['trip_duration'],ax=ax1)
box2=sn.boxplot(x=train[train['store_and_fwd_flag']=='Y']['trip_duration'],ax=ax2)
#box1.set(xlim=(0,100000))               
#box2.set(xlim=(0,100000))
train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'])
train['month']=train['pickup_datetime'].dt.month
train['weekday']=train['pickup_datetime'].dt.weekday
train['pickup_hour']=train['pickup_datetime'].dt.hour
train['pickup_minute'] = train['pickup_datetime'].dt.minute

test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])
test['month']=test['pickup_datetime'].dt.month
test['weekday']=test['pickup_datetime'].dt.weekday
test['pickup_hour']=test['pickup_datetime'].dt.hour
test['pickup_minute'] = test['pickup_datetime'].dt.minute
df=train.groupby('pickup_hour')['trip_duration'].mean().reset_index()
sn.barplot(x='pickup_hour',y='trip_duration',data=train)
train.info()
categorical_vars=['vendor_id','pickup_hour','pickup_minute','weekday','month','store_and_fwd_flag','passenger_count']
for var in categorical_vars:
    train=pd.concat([train,pd.get_dummies(train[var],prefix=var)],1)
    train=train.drop(var,1)
    test=pd.concat([test,pd.get_dummies(test[var],prefix=var)],1)
    test=test.drop(var,1)
train=train.drop(['pickup_datetime','dropoff_datetime','pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude','id','trip_duration'],1)
test=test.drop(['pickup_datetime','pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude','id'],1)
s=[]
for col in train.columns:
    if col not in test.columns:
        s.append(col)
s
train=train.drop(['passenger_count_7', 'passenger_count_8'],1)
train.shape,test.shape
from sklearn.model_selection import train_test_split
train, valid = train_test_split(train, test_size = 0.2)

y_train=train['log_trip_duration']
train=train.drop('log_trip_duration',1)
y_valid=valid['log_trip_duration']
valid=valid.drop('log_trip_duration',1)
y_train=y_train.reset_index().drop('index',axis = 1)
y_valid=y_valid.reset_index().drop('index',axis = 1)
def rmsle(y,y_):
    logy_=np.nan_to_num(np.array([np.log(p+1) for p in y_]))
    calc=(y-logy_)**2
    return np.sqrt(np.mean(calc))
from sklearn.metrics import make_scorer
rmsle_scorer=make_scorer(rmsle,greater_is_better=False)
import xgboost as xgb
dtrain = xgb.DMatrix(train, label=y_train)
dvalid = xgb.DMatrix(valid, label=y_valid)
dtest = xgb.DMatrix(test)
watchlist = [(dtrain, 'train'),(dvalid, 'valid')]

params = {'min_child_weight': 20, 'eta': 0.1, 'colsample_bytree': 0.3, 'max_depth': 10,
'subsample': 0.3, 'lambda': 6, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear', 'feval':rmsle_scorer}
model = xgb.train(
    params,
    dtrain,
    1000,
    evals=watchlist,
    maximize=False,
    verbose_eval=1,
    early_stopping_rounds=50)

#pred=model.predict(train)
#rmsle(y_train,pred)
pred = model.predict(dtest)
pred = np.exp(pred) - 1
submission = pd.concat([test_id, pd.DataFrame(pred)], axis=1)
submission.columns = ['id','trip_duration']
submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)
submission.to_csv("submission.csv", index=False)
