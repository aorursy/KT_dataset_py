import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv('../input/sales_train.csv')
test=pd.read_csv('../input/test.csv')
data.head()
test.head()
data['date'] = data['date'].apply(lambda x: datetime.strptime(x,'%d.%m.%Y'))
print(data.dtypes)

data['date_block_num'] = data['date_block_num'].astype(str)

data['shop_id'] = data['shop_id'].astype(str)

data['item_id'] = data['item_id'].astype(str)

print(data.dtypes)
data.describe()
data.apply(lambda x:sum(x.isnull()),axis=0)
data.boxplot(column = 'item_price')

plt.show()
data["shop_id"].unique()
data["item_id"].unique()
data["date_block_num"].unique()
data['shop_id'].value_counts().plot(kind='bar',figsize=(15, 5))
data['date_block_num'].value_counts().plot(kind='bar',figsize=(15, 5))
#data['item_id'].value_counts().plot(kind='bar',figsize=(15, 5))

data['item_id'].value_counts()
modified = data.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

train_df = modified.reset_index()

train_df['shop_id']= train_df.shop_id.astype('str')

train_df['item_id']= train_df.item_id.astype('str')

train_df.head()
train_df = train_df[['shop_id', 'item_id','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33']]

train_df.head()
X_train=train_df.iloc[:,  (train_df.columns != '33')].values

y_train=train_df.iloc[:, train_df.columns == '33'].values
rf = RandomForestRegressor(random_state = 10)

# Train the model on training data

rf.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

rmse_dmy = np.sqrt(mean_squared_error(y_train, rf.predict(X_train)))

print('RMSE: %.4f' % rmse_dmy)
param_grid = {

    'bootstrap': [True],

    'max_depth': [5,10],

    #'max_features': [20],

    'min_samples_leaf': [3],

    'min_samples_split': [8],

    'n_estimators': [100]

}

# Create a based model

rf = RandomForestRegressor()



grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)



rmse_dmy = np.sqrt(mean_squared_error(y_train, grid_search.predict(X_train)))

print('RMSE: %.4f' % rmse_dmy)
import xgboost as xgb

param = {'max_depth':12,

         'subsample':1,  

         'min_child_weight':0.5,  

         'eta':0.3,

         'num_round':1000, 

         'seed':42,  

         'silent':0,

         'eval_metric':'rmse',

         'early_stopping_rounds':100

        }



progress = dict()

xgbtrain = xgb.DMatrix(X_train, y_train)

watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)

preds = bst.predict(xgb.DMatrix(X_train))

rmse_dmy = np.sqrt(mean_squared_error(y_train,preds))

print('RMSE: %.4f' % rmse_dmy)
preds_XG = bst.predict(xgb.DMatrix(X_train))

preds_RFCV = grid_search.predict(X_train)



## Stacking 

Stacking_data_Train=pd.DataFrame( {'RandomForest':preds_RFCV,'CGB':preds_XG})



param = {'max_depth':12,

         'subsample':1,  

         'min_child_weight':0.5,  

         'eta':0.3,

         'num_round':1000, 

         'seed':42,  

         'silent':0,

         'eval_metric':'rmse',

         'early_stopping_rounds':100

        }



progress = dict()

xgbtrain_2 = xgb.DMatrix(Stacking_data_Train, y_train)

watchlist_2  = [(xgbtrain_2,'train-rmse')]

bst2 = xgb.train(param, xgbtrain_2)

preds = bst2.predict(xgb.DMatrix(Stacking_data_Train))

rmse_dmy = np.sqrt(mean_squared_error(y_train,preds))

print('RMSE: %.4f' % rmse_dmy)
test_df = test.copy()

test_df['shop_id']= test_df.shop_id.astype('str')

test_df['item_id']= test_df.item_id.astype('str')



test_df = test_df.merge(train_df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

test_df.head()

test_df.columns =['ID','shop_id','item_id','-1','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']

test_df.head()
X_test = test_df.iloc[:, (test_df.columns != 'ID') & (test_df.columns != '-1')].values
preds_XG = bst.predict(xgb.DMatrix(X_test))

preds_RFCV = grid_search.predict(X_test)



## Stacking 

Stacking_data_Test=pd.DataFrame( {'RandomForest':preds_RFCV,'CGB':preds_XG})

preds_vf = bst2.predict(xgb.DMatrix(Stacking_data_Test))
preds_vf = list(map(lambda x: min(20,max(x,0)), list(preds_vf)))

final = pd.DataFrame({'ID':test_df.ID,'item_cnt_month': preds_vf })
final.to_csv('Stacking_RFCV_XGB.csv',index=False)