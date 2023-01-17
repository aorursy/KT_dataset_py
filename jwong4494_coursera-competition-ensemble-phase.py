import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.sparse 
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm_notebook

from itertools import product

# Any results you write to the current directory are saved as output.
path = '../input/coursera-competition-eda-phase/'
all_data = pd.read_csv(path + 'all_data.csv')
test = pd.read_csv(path + 'test.csv')
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df
all_data = downcast_dtypes(all_data)
test = downcast_dtypes(test)
all_data.index.values
all_data.head()
all_data.drop(['Unnamed: 0'], axis=1, inplace=True)
# all_data.shop_id = all_data.shop_id.astype('str')
# all_data.item_id = all_data.item_id.astype('str')
# all_data.item_category_id = all_data.item_category_id.astype('str')
test.drop(['Unnamed: 0'], axis=1, inplace=True)
# test.shop_id = test.shop_id.astype('str')
# test.item_id = test.item_id.astype('str')
# test.item_category_id = test.item_category_id.astype('str')
test.head()
all_data.head()
testID = test.ID
meta_cat_dict = {key:idx for idx, key 
                 in enumerate(all_data.meta_cat.unique())}

all_data['meta_cat_num'] = all_data['meta_cat'].map(meta_cat_dict)
test['meta_cat_num'] = test['meta_cat'].map(meta_cat_dict)
dates = all_data['date_block_num']
last_block = dates.max()
print('Test `date_block_num` is %d' % last_block)
to_drop_cols = ['target', 'date_block_num', 'meta_cat']

dates_train = dates[dates <  last_block]
dates_test  = dates[dates == last_block]

X_train = all_data.loc[dates <  last_block].drop(to_drop_cols, axis=1)
X_test =  all_data.loc[dates == last_block].drop(to_drop_cols, axis=1)

y_train = all_data.loc[dates <  last_block, 'target'].values
y_test =  all_data.loc[dates == last_block, 'target'].values
# Linear Regression
lr = LinearRegression()
lr.fit(X_train.values, y_train)
pred_lr = lr.predict(X_test.values)

print('Test R-squared for linreg is %f' % r2_score(y_test, pred_lr))

# LightGBM
lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':1, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0 
              }

model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)
pred_lgb = model.predict(X_test)

print('Test R-squared for LightGBM is %f' % r2_score(y_test, pred_lgb))


# Testing for XGBoost

#all_data.shop_id = all_data.shop_id.astype('str')
#all_data.item_id = all_data.item_id.astype('str')
#all_data.item_category_id = all_data.item_category_id.astype('str')

to_drop_cols = ['target', 'date_block_num', 'meta_cat']

dates_train = dates[dates <  last_block]
dates_test  = dates[dates == last_block]

X_train = all_data.loc[dates <  last_block].drop(to_drop_cols, axis=1)
X_test =  all_data.loc[dates == last_block].drop(to_drop_cols, axis=1)

y_train = all_data.loc[dates <  last_block, 'target'].values
y_test =  all_data.loc[dates == last_block, 'target'].values
# XGBoost
import xgboost as xgb
xgb_param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}

#xgbtrain = xgb.DMatrix(X_train)
xgb_model = xgb.train(xgb_param, xgb.DMatrix(X_train, label=y_train), 100)
X_test = xgb.DMatrix(X_test)
pred_xgb = xgb_model.predict(X_test)
print('Test R-squared for XGBoost is %f' % r2_score(y_test, pred_xgb))
test.drop(['meta_cat', 'ID'], axis=1, inplace=True)
test['item_target_enc'] = test['item_id'].map(all_data['item_target_enc'])
test = test[['shop_id', 'item_id', 'item_target_enc', 'item_category_id', 'meta_cat_num']]
test = xgb.DMatrix(test)
pred_xgb = xgb_model.predict(test)
pred = list(map(lambda x: min(20,max(x,0)), list(pred_xgb)))
submission = pd.DataFrame({'ID':testID,'item_cnt_month': pred })

submission.to_csv('submission.csv', index=False)

