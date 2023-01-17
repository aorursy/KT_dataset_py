import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns
sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test.info()
test.head()
sales.info()
sales.head()
sales['item_price'].max()
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=sales.item_cnt_day)
plt.figure(figsize=(10,4))
plt.xlim(sales.item_price.min(), sales.item_price.max()*1.1)
sns.boxplot(x=sales.item_price)

sales.drop(sales[sales['item_cnt_day']<0].index,axis=0,inplace=True)

subset  = ['date','date_block_num','shop_id','item_id', 'item_cnt_day']
sales.drop_duplicates(subset=subset , inplace=True)

max_price=sales['item_price'].max()
most_expensive_item=sales.loc[sales['item_price']==max_price,'item_id'].values[0]
sales.drop(sales['item_price'].idxmax(),axis=0,inplace=True)
del max_price, most_expensive_item
sales.drop(sales['item_price'].idxmin(),axis=0,inplace=True)
sales=sales.iloc[:,0:7]
sales = sales[sales.item_price<100000]
sales = sales[sales.item_cnt_day<1001]
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=sales.item_cnt_day)
plt.figure(figsize=(10,4))
plt.xlim(sales.item_price.min(), sales.item_price.max()*1.1)
sns.boxplot(x=sales.item_price)
sales.head()
agg_df = sales.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index()
agg_df.columns = ['date_block_num','shop_id','item_id','item_cnt_day']
agg_df['item_cnt_day'].clip(0,20,inplace=True)

exc_item_cnt=agg_df.iloc[:,:-1]
item_cnt=agg_df.iloc[:,-1:]
exc_item_cnt
item_cnt
x_train, x_test,y_train,y_test = train_test_split(exc_item_cnt,item_cnt,test_size=0.33, random_state=14)
from xgboost import XGBRegressor
xgb_model = XGBRegressor(random_state=14, colsample_bylevel=1,
                         colsample_bytree=0.5, learning_rate=0.2, seed=42, max_depth=5,
                         n_estimators=250, min_child_weight=250, subsample=0.8)
xgb_model.fit(x_train,y_train,eval_metric="rmse", 
              eval_set=[(x_train, y_train), (x_test, y_test)], 
              verbose=True, 
              early_stopping_rounds=15)

y_pred = xgb_model.predict(x_test)
y_pred = y_pred.tolist()
xgb_r2=r2_score(y_test,y_pred)
xgb_rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
AdaBoostRegressor
ada_model = AdaBoostRegressor(random_state=14,n_estimators=250,learning_rate=0.3)

ada_model.fit(x_train,y_train)
y_pred = ada_model.predict(x_test)
ada_rmse=np.sqrt(mean_squared_error(y_test,y_pred))
ada_r2=r2_score(y_test,y_pred)
print("R2 Score:",r2_score(y_test,y_pred))
print('Root Mean Squared Error :', np.sqrt(mean_squared_error(y_test,y_pred)))
rf = RandomForestRegressor(n_estimators = 50,random_state=14)

rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
rf_rmse=np.sqrt(mean_squared_error(y_test,y_pred))
rf_r2=r2_score(y_test,y_pred)
print("R2 Score:",r2_score(y_test,y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error :', np.sqrt(mean_squared_error(y_test,y_pred)))
import lightgbm as lgb
params = {
    'task': 'train','boosting_type': 'gbdt','objective': 'regression','metric': 'rmse',
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 10,
    "max_depth": 8,
    "num_leaves": 128,  
    "max_bin": 512,
    "num_iterations": 40,
    "n_estimators": 250
}

lgb_model = lgb.LGBMRegressor(**params)

lgb_model.fit(x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1000)
y_pred = lgb_model.predict(x_test)
light_r2=r2_score(y_test,y_pred)
light_rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
RMSE = [xgb_rmse, ada_rmse, rf_rmse, light_rmse]
import seaborn as sns 
import matplotlib.pyplot as plt
y_ax = ['XGBoost' ,'AdaBoost', 'Random Forest Regression','Lightgbm']
x_ax = RMSE

sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.1")
plt.xlabel('RMSE')
R2 = [xgb_r2, ada_r2, rf_r2, light_r2]
import seaborn as sns 
import matplotlib.pyplot as plt
y_ax = ['XGBoost' ,'AdaBoost', 'Random Forest Regression','Lightgbm']
x_ax = R2

sns.barplot(x=x_ax,y=y_ax,linewidth=1.5,edgecolor="0.1")
plt.xlabel('R2')
t_p=rf.predict(test)
p_df=pd.DataFrame(t_p,columns=['item_cnt_month',])
p_df=p_df.clip(0,20)
submission.drop(columns ="item_cnt_month",inplace = True)
result=pd.concat([submission,p_df],axis=1)
result.head(6)
result.to_csv('submission.csv', index=False)
