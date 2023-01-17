import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from xgboost import XGBClassifier

df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")
df.head()
split = "2017-01-01"
df['date'] =  pd.to_datetime(df['date'])
df = df.set_index('date')
df.head()
df_XG = df.groupby('store').resample('M')['sales'].sum()
df_XG_store_item = df.groupby(['store','item'])['sales'].resample('M').sum()
df_XG.head()
df_XG_store_item.tail()
df_XG.head()
df_XG_store_item.head()
df_XG_store_item.tail()
df_XG = df_XG.reset_index()
df_XG.head()
df_XG_store_item= df_XG_store_item.reset_index() 
df_XG_store_item.head()
df_XG = df_XG.set_index('date')
df_XG_store_item = df_XG_store_item.set_index('date')
df_XG_store_item.tail()
df_XG.info()
df_XG['month'] =df_XG.index.month
df_XG['year'] = df_XG.index.year
df_XG_store_item['month'] =df_XG_store_item.index.month
df_XG_store_item['year'] = df_XG_store_item.index.year
df_XG.head()
df_train = df_XG[ :split ] 
df_test = df_XG[split : ] 
df_train_SI = df_XG_store_item[ :split ] 
df_test_SI = df_XG_store_item[split : ] 
y_train = df_train.loc[:,'sales']
y_test= df_test.loc[:,'sales']
X_train = df_train.drop (['sales'],axis=1)
X_test = df_test.drop (['sales'],axis=1)
y_train_SI = df_train_SI.loc[:,'sales']
y_test_SI= df_test_SI.loc[:,'sales']
X_train_SI = df_train_SI.drop (['sales'],axis=1) 
X_test_SI = df_test_SI.drop (['sales'],axis=1)
y_train_SI.tail()
X_test_SI.tail()
df_XG_month= df.resample('M')['sales'].sum()
df_XG_month = df_XG_month.reset_index()
df_XG_month['year'] = df_XG_month['date'].dt.year
df_XG_month = df_XG_month.set_index('date')
df_train_month = df_XG_month[ :split ] 
df_test_month = df_XG_month[split : ] 
df_train_month.head()
y_train_month = df_train_month.loc[:,'sales']
y_test_month= df_test_month.loc[:,'sales']
X_train_month = df_train_month.drop (['sales'],axis=1) 
X_test_month = df_test_month.drop (['sales'],axis=1)
XG_model_month = xgb.XGBRegressor(n_estimators=1000)
XG_model_month.fit(X_train_month, y_train_month,eval_set=[(X_test_month, y_test_month)], early_stopping_rounds=50,verbose=False) # Change verbose to True to see it train
_ = plot_importance(XG_model_month, height=0.9)
XG_test_prediction = XG_model_month.predict(X_test_month)
print(XG_test_prediction)
# You can see by looking at the results that this is a terrible forecast as there are not enough features
XG_model = xgb.XGBRegressor(n_estimators=1000)
X_test.head()
y_test.head()
%%time
XG_model.fit(X_train, y_train,eval_set=[(X_test, y_test)], early_stopping_rounds=50,verbose=False) # Change verbose to True to see it train
_ = plot_importance(XG_model, height=0.9)
XG_test_prediction = XG_model.predict(X_test)
XG_test_all =X_test.copy()
XG_train_all =X_train.copy()
XG_test_all['XG prediction'] = XG_model.predict(X_test)
XG_train_all['XG prediction'] =XG_model.predict(X_train)
XG_test_all['sales'] = y_test
XG_train_all['sales'] = y_train
df_xg_all = pd.concat([XG_test_all, XG_train_all], sort=False)
XG_test_all.sum()
X_train.head()
df_xg_all.head()
y_train.head()
df_xg_all.sample(10)
XG_test_all.describe()
XG_train_all.describe()
XG_test_all['sales'].sum()
XG_test_all['XG prediction'].sum()
XG_RMSE  = np.mean(np.sqrt((XG_test_all['XG prediction'] - XG_test_all['sales']) ** 2))  
print(XG_RMSE)
XG_test_all.shape
XG_test_all.head()
df_xg_all.describe()
_ = df_xg_all[['sales','XG prediction']].plot(figsize=(15, 5))
# as too many options the plot is not very useful
df_xg_all_1 = df_xg_all[(df_xg_all.store==1)]
df_xg_all_3 = df_xg_all[(df_xg_all.store==3)]
XG_train_all_1 = XG_train_all[(XG_train_all.store==1)]
_ = df_xg_all_1[['sales','XG prediction']].plot(figsize=(15, 5))
_ = df_xg_all_3[['sales','XG prediction']].plot(figsize=(15, 5))
df_xg_all.sample(10)
df_xg_all.head()
X_test_SI.head()
XG_model_SI = xgb.XGBRegressor(n_estimators=1000)
%%time
XG_model_SI.fit(X_train_SI, y_train_SI,eval_set=[(X_test_SI, y_test_SI)], early_stopping_rounds=50,verbose=False) # Change verbose to True to see it train
#XG_model.fit(X_train, y_train,eval_set=[(X_test, y_test)]
_ = plot_importance(XG_model_SI, height=0.9)
XG_test_prediction_SI = XG_model_SI.predict(X_test_SI)
XG_test_prediction_SI.sum()
XG_SI_test_all =X_test_SI.copy()
XG_SI_train_all =X_train_SI.copy()
XG_SI_test_all['XG prediction'] = XG_model_SI.predict(X_test_SI)
XG_SI_test_all['sales'] = y_test_SI
XG_SI_train_all['sales'] = y_train_SI
XG_SI__RMSE  = np.mean(np.sqrt((XG_SI_test_all['XG prediction'] - XG_SI_test_all['sales']) ** 2)) 
print(XG_SI__RMSE) # This result is very good
XG_SI_test_all.shape
df_xg_SI_all = pd.concat([XG_SI_test_all, XG_SI_train_all], sort=False)
XG_SI_test_all.sample(10)
XG_SI_train_all.sample(3)
df_xg_SI_all.sample(5)
_ = df_xg_SI_all[['sales','XG prediction']].plot(figsize=(15, 5))
df_xg_SI_all_1_1 = df_xg_SI_all[(df_xg_SI_all.store==1)&(df_xg_SI_all.item==1)]
_ = df_xg_SI_all_1_1[['sales','XG prediction']].plot(figsize=(15, 5))
df_xg_SI_all_1_2 = df_xg_SI_all[(df_xg_SI_all.store==1)&(df_xg_SI_all.item==2)]
_ = df_xg_SI_all_1_2[['sales','XG prediction']].plot(figsize=(15, 5))
df_xg_SI_all_2_2 = df_xg_SI_all[(df_xg_SI_all.store==2)&(df_xg_SI_all.item==2)]
_ = df_xg_SI_all_2_2[['sales','XG prediction']].plot(figsize=(15, 5))
df_xg_SI_all.nunique()
df_xg_SI_all.store.value_counts()
