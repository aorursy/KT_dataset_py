!pip install catboost -q
import numpy as np  
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# LOAD DATA
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
# Add date_block_num to test
# Identify the next date_block_num for the test set
test_dates = train.groupby(['shop_id'])['date_block_num'].max().reset_index()
test_dates['date_block_num'] += 1
#test_dates['date_block_num'] = 34
test = test.merge(test_dates,how='left',on='shop_id')


# Similar shop names but different shop_ids - > rewrite shopids
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
# Number of items available in test, but not in training
test_shops = test.shop_id.unique().tolist()

missing_items = list(np.setdiff1d(test.item_id.unique().tolist(),train.item_id.unique().tolist()))
missing_rows = test[test.item_id.isin(missing_items)].shape[0]
print('Number of items not available in training: ',len(missing_items))
print('Number of rows not available in training: ', missing_rows)
print('Number of rows for testing: ', test.shape[0])
print('Percentage missing: ', missing_rows * 100 / test.shape[0])
# identify shop item combination across train and test
train_shop_item=train[['shop_id','item_id']].drop_duplicates()
test_shop_item=test[['shop_id','item_id']].drop_duplicates()

#combine shop item combinations
shop_item = pd.concat([train_shop_item,test_shop_item]).drop_duplicates()

# extract dateblocks from train
train_shop_dates = train[['shop_id','date_block_num']].drop_duplicates()

full=train_shop_dates.merge(shop_item,
                      how='left',
                      on='shop_id')

merged=full.merge(train,
          how='left',
          on=['shop_id','item_id','date_block_num'])

train = merged.loc[:,['shop_id','item_id','date','date_block_num','item_price','item_cnt_day']]
train.fillna(0,inplace=True)
items_full=items.merge(item_categories,how='left',on=['item_category_id'])
missing=test[test.item_id.isin(missing_items)].merge(items_full,how='left',on='item_id')

# Generate a replacement item_id for items found in test, but missing in training
# Replacement strategy : most popular item within that category for that particular shop

item_cat_map = items[items.item_id.isin(missing_items)][['item_id','item_category_id']]
item_cat_map_dict = dict(zip(item_cat_map.item_id.values,item_cat_map.item_category_id.values))


train_monthly=train.groupby(['shop_id','item_id','date_block_num'])\
                   .agg(item_sum_mth=pd.NamedAgg(column='item_cnt_day',aggfunc='sum')).reset_index()\
                   .merge(items_full[['item_id','item_category_id']],
                          how='left',
                          on='item_id')


most_pop_item_per_shop_and_cat=train_monthly.groupby(['shop_id','item_category_id'])['item_id']\
                                            .apply(lambda x : x.value_counts().head(1))\
                                            .reset_index(name='occurences')\
                                            .rename(columns={'level_2' : 'item_id'})

most_pop_item_per_shop_and_cat['item_id'] = most_pop_item_per_shop_and_cat['item_id'].astype('int64')


# Assumption new items will sell similar to the most popular item within the same category sold at the same shop
replacement=missing.merge(most_pop_item_per_shop_and_cat,
                          how='left',
                          on=['shop_id','item_category_id'], 
                          suffixes=('_original','_replacement'))

# Replacement kind of work, but it seems that some shops are selling items in a category they havent done before
print(replacement[replacement.item_id_replacement.isna()].shape,replacement.shape)
print('\n')

train_shop_cat_map = train_monthly.groupby(['shop_id'])['item_category_id'] \
                                     .apply(lambda x: x.unique().astype('int16').tolist())\
                                     .to_dict()
test_shop_cat_map = test.merge(items[['item_id','item_category_id']],how='left',on='item_id')\
                        .groupby(['shop_id'])['item_category_id']\
                        .apply(lambda x: x.unique().astype('int16').tolist())\
                        .to_dict()

# Check whats not matching up in train and test
result = {}
for shop, item_cat in test_shop_cat_map.items():
    #get item categories that are not in training but in test
    result[shop] = list(np.setdiff1d(item_cat,train_shop_cat_map[shop]))

#import pprint 
#pprint.pprint(result)

# Lets use historical data from other shops on scenarios where a shop has started selling a new item in a new category
replacement_no_nan=missing.merge(most_pop_item_per_shop_and_cat,
                                 how='left',
                                 on=['item_category_id'], 
                                 suffixes=('_original','_replacement'))
replacement.update(replacement_no_nan,overwrite=False)
replacement['item_id_replacement'] = replacement['item_id_replacement'] .astype('int16')
replacement['occurences'] = replacement['occurences'] .astype('int16')

# 1 add inn replacement item_ids
cols = ['ID','shop_id','item_id_original','item_id_replacement']

test = test.merge(replacement[cols],
                        how='left',
                        left_on=['ID','shop_id','item_id'],
                        right_on=['ID','shop_id','item_id_original'],
                        suffixes=('','_repl'))
# 
test['item_id_new'] = np.where(test.item_id_replacement.isna(),
                               test.item_id,
                               test.item_id_replacement)

test['item_id_new'] = test['item_id_new'].astype('int16')

test.rename(columns={'item_id' : 'item_id_org'},inplace=True)
test.rename(columns={'item_id_new' : 'item_id'},inplace=True)
test.drop(columns=['item_id_original','item_id_org','item_id_replacement'],inplace=True)
# Preliminary data cleaning
train = train[(train.item_price <100000) & (train.item_price > 0)]
# Generate monthly aggregates
agg_train=train.groupby(['shop_id','item_id','date_block_num'])\
               .agg(
                            max_price=pd.NamedAgg(column='item_price', aggfunc='max'), 
                            min_price=pd.NamedAgg(column='item_price', aggfunc='min'),
                            mean_price=pd.NamedAgg(column='item_price', aggfunc='mean'),
                            item_sum_mth=pd.NamedAgg(column='item_cnt_day',aggfunc='sum'),
                            item_mean_mth=pd.NamedAgg(column='item_cnt_day',aggfunc='mean'),
                            item_std_mth=pd.NamedAgg(column='item_cnt_day', aggfunc=np.nanstd),
                            num_days_sales=pd.NamedAgg(column='date',aggfunc='nunique')
                   )

agg_train = agg_train.reset_index()
# Add dmy columns (so we may concat dataframes to ease preprocessing)
agg_train['CATEGORY'] = 'TRAIN'
agg_train['ID'] = -1
test['max_price'] = 0
test['min_price'] = 0
test['mean_price'] = 0
test['item_sum_mth'] = 0
test['item_mean_mth'] = 0
test['item_std_mth'] = 0
test['CATEGORY'] = 'TEST'
test['num_days_sales'] = 0

# Combine dataframes to ease feature engineering 
agg_combined = pd.concat([agg_train,test])
# Downcast
agg_combined['shop_id'] = agg_combined['shop_id'].astype('int32')
agg_combined['item_id'] = agg_combined['item_id'].astype('int32')
agg_combined['date_block_num'] = agg_combined['date_block_num'].astype('int32')

agg_combined['max_price'] = agg_combined['max_price'].astype('float32')
agg_combined['min_price'] = agg_combined['min_price'].astype('float32')
agg_combined['mean_price'] = agg_combined['mean_price'].astype('float32')

agg_combined['item_sum_mth'] = agg_combined['item_sum_mth'].astype('int32')
agg_combined['item_mean_mth'] = agg_combined['item_mean_mth'].astype('float32')
agg_combined['item_std_mth'] = agg_combined['item_std_mth'].astype('float32')
agg_combined['date_block_num_prev']=agg_combined.sort_values(['shop_id','item_id','date_block_num']).groupby(['shop_id','item_id'])['date_block_num'].shift(-1).values
agg_combined['date_block_num_prev'].fillna(0,inplace=True)
agg_combined['mth_since_l_sale'] = agg_combined['date_block_num'] - agg_combined['date_block_num_prev']
agg_combined.sort_values(['shop_id','item_id','date_block_num'],inplace=True)
agg_combined['item_cum_sum']=agg_combined.groupby(['shop_id','item_id']).cumsum()['item_sum_mth']

agg_combined['total_item_trend'] = agg_combined.groupby(['item_id','date_block_num'])['item_sum_mth'].transform(sum)
agg_combined['totel_item_price_trend'] = agg_combined.groupby(['item_id','date_block_num'])['mean_price'].transform(np.mean)
agg_combined['totel_item_price_trend_std'] = agg_combined.groupby(['item_id','date_block_num'])['mean_price'].transform(np.std)


agg_combined['date_block_num']+=1

agg_combined['first_month_sale'] = agg_combined.groupby(['shop_id','item_id'])['date_block_num'].transform(min)
agg_combined['month_since_first_sale'] = agg_combined.date_block_num - agg_combined.first_month_sale

# add some dateblock normalized features
agg_combined['total_item_trend_N'] = agg_combined['total_item_trend']  / agg_combined['date_block_num'] 
agg_combined['item_sum_mth_N'] = agg_combined['item_sum_mth'] / agg_combined['date_block_num'] 
agg_combined['item_cum_sum_N'] = agg_combined['item_cum_sum'] / agg_combined['date_block_num']
agg_combined['cum_sum_price_N'] = agg_combined.groupby(['shop_id','item_id']).cumsum()['mean_price'] / agg_combined['date_block_num']
agg_combined['item_mean_price_N'] = (agg_combined['mean_price'] - agg_combined['totel_item_price_trend']) / agg_combined['totel_item_price_trend']
agg_combined['item_mean_price_N'].fillna(0,inplace=True)


#agg_combined['mean_item_target']=agg_combined.groupby(['shop_id','item_id'])['item_sum_mth'].mean()

#agg_train['item_sum_returns'] = train[train.item_cnt_day < 0].groupby(['shop_id','item_id','date_block_num'])['item_cnt_day'].sum()
#agg_train['item_sum_returns'].fillna(0,inplace=True)
agg_combined['item_revenue_mth'] = agg_combined['mean_price'] * agg_combined['item_sum_mth']
                                                                        

# Lagged features
agg_combined_lagged = agg_combined

def create_lag(df,column, lags):
    
    for lag in lags:
        df[column + '_lagged_'+str(lag)] = df.sort_values(['shop_id','item_id','date_block_num'])\
                                  .groupby(['shop_id','item_id',])[column].shift(-lag)
        df[column + '_lagged_' +str(lag)].fillna(0,inplace=True)
    return df

agg_combined_lagged = create_lag(agg_combined_lagged,'item_sum_mth',[1,2,3,12])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_cum_sum',[1,2,3,12])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_std_mth',[1,12])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_mean_mth',[1,12])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_std_mth',[1,12])
agg_combined_lagged = create_lag(agg_combined_lagged,'num_days_sales',[1,12])
agg_combined_lagged = create_lag(agg_combined_lagged,'mean_price',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_revenue_mth',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_cum_sum',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'cum_sum_price_N',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_sum_mth_N',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'cum_sum_price_N',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'total_item_trend_N',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'month_since_first_sale',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'mth_since_l_sale',[1])
agg_combined_lagged = create_lag(agg_combined_lagged,'item_mean_price_N',[1])

agg_combined_lagged['sum_cum_sum_ratio_lagged_1'] = agg_combined_lagged['item_sum_mth_N_lagged_1'] / agg_combined_lagged['cum_sum_price_N_lagged_1']
agg_combined_lagged['sum_cum_sum_ratio_lagged_1'].fillna(0,inplace=True)


agg_combined_lagged = agg_combined_lagged.reset_index()

agg_combined_lagged.sort_values(['shop_id','item_id','date_block_num'])


data = agg_combined_lagged.merge(items[['item_id','item_category_id']],
                                 how='left',
                                 on='item_id')

# Add month + year
data['month'] = (data['date_block_num'] % 12) +1
data['year'] = round(data['date_block_num'] / 12)


# Select feature columns
all_columns=data.columns.tolist()
lagged_columns = [x for x in all_columns if 'lagged' in x ]
categorical_columns = ['shop_id','item_id','item_category_id','month','year']
numeric_columns = ['date_block_num']

target = ['item_sum_mth']
features = categorical_columns + lagged_columns  + numeric_columns

data = data[features+target + ['CATEGORY', 'ID']]

# clip target
data[target] = data[target].clip(0,20)
# split into the appropiate datasets
sub_cols = categorical_columns + lagged_columns + numeric_columns + ['ID']
submission = data.loc[data.CATEGORY=='TEST',sub_cols]

data_cols = categorical_columns + lagged_columns + numeric_columns + target
train_new = data.loc[data.CATEGORY=='TRAIN',data_cols]
def train_test_val_split_date(df,features,target, val_block, test_block):
    train_x = df.loc[df.date_block_num < val_block,features]
    train_y = df.loc[df.date_block_num < val_block,target]
    
    val_x = df.loc[(df.date_block_num >= val_block ) &(df.date_block_num < test_block),features]
    val_y = df.loc[(df.date_block_num >= val_block ) &(df.date_block_num < test_block),target]
    
    
    test_x = df.loc[df.date_block_num >= test_block,features]
    test_y = df.loc[df.date_block_num >= test_block,target]
    
    return train_x, test_x,val_x, train_y, test_y, val_y


train_x,test_x,val_x, train_y,test_y, val_y = train_test_val_split_date(train_new, features, target, 32,33)


train_x.shape, test_x.shape,val_x.shape
train_x[categorical_columns] = train_x[categorical_columns].astype('str')
test_x[categorical_columns] = test_x[categorical_columns].astype('str')
val_x[categorical_columns] = val_x[categorical_columns].astype('str')
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor

catboost_params = {'objective':'poisson',
                    'iterations' : 1000,
                   'depth' : 10,
                   'learning_rate' : 0.2,
                   'bagging_temperature':0.2,
                   'l2_leaf_reg' : 9,
                   'loss_function' : 'RMSE',
                  'task_type':'GPU',
                   'early_stopping_rounds':20,
                  'max_ctr_complexity':1}


train_pool = Pool(train_x, train_y, cat_features=categorical_columns)
val_pool = Pool(val_x, val_y, cat_features=categorical_columns)
test_pool = Pool(test_x, test_y,cat_features=categorical_columns) 

cat_model = CatBoostRegressor(**catboost_params)
cat_model.fit(train_pool,
             eval_set=val_pool)
cat_preds = np.clip(cat_model.predict(test_pool),0,20)

print('RMSE on test: ', np.sqrt(mean_squared_error(test_y,cat_preds)))
# From https://www.kaggle.com/ashishpatel26/feature-importance-of-lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl

mpl.rcParams['text.color'] = 'w'
mpl.rcParams['xtick.color'] = 'w'
mpl.rcParams['ytick.color'] = 'w'
mpl.rcParams['axes.labelcolor'] = 'w'

feat_imp=pd.DataFrame(sorted(zip(cat_model.feature_importances_,train_x.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 5))
sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values(by="Value", ascending=False))
plt.title('Catboost Features')
plt.tight_layout()
plt.show()
corr=data[features+target].corr()
corr.style.background_gradient(cmap='coolwarm')
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor

catboost_params = {'iterations' : 1500,
                   'depth' : 10,
                   'learning_rate' : 0.3,
                   'bagging_temperature':0.2,
                   'l2_leaf_reg' : 9,
                   'loss_function' : 'RMSE',
                  'task_type':'GPU',
                   'early_stopping_rounds':20,
                  'max_ctr_complexity':1}
cat_model_full = CatBoostRegressor(**catboost_params)

X = data[features]
Y = data[target]
X[categorical_columns] = X[categorical_columns].astype('str')
full = Pool(X,Y,cat_features=categorical_columns)
cat_model_full.fit(full)
import lightgbm as lgb

lgbm_params={'learning_rate': 0.2,
        'objective':'poisson',
        'metric':'rmse',
        'verbose': 1,
        'random_state':42       }


train_x[categorical_columns] = train_x[categorical_columns].astype('category')
val_x[categorical_columns] = val_x[categorical_columns].astype('category')
test_x[categorical_columns] = test_x[categorical_columns].astype('category')

lgbm_model = lgb.LGBMRegressor(**lgbm_params, n_estimators=1000, categorical_feature='auto')
lgbm_model.fit(train_x, train_y, eval_set=[(val_x, val_y)], early_stopping_rounds=50, verbose=10)

lgbm_preds = lgbm_model.predict(test_x)
print('RMSE on test: ', np.sqrt(mean_squared_error(test_y,np.clip(lgbm_preds,0,20))))
# From https://www.kaggle.com/ashishpatel26/feature-importance-of-lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

feat_imp=pd.DataFrame(sorted(zip(lgbm_model.feature_importances_,train_x.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 5))
sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
# Train full
X = data[features]
X[categorical_columns] = X[categorical_columns].astype('category')
Y=data[target]
import lightgbm as lgb

lgbm_params={'learning_rate': 0.2,
        'objective':'poisson',
        'metric':'rmse',
        'random_state':42,
        'device_type' :'cpu',
             'nthread':-1
       }

lgbm_model_full = lgb.LGBMRegressor(**lgbm_params, n_estimators=1500)
lgbm_model_full.fit(X, Y,categorical_feature='auto')
train_x[categorical_columns]=train_x[categorical_columns].astype('float32')
val_x[categorical_columns]=val_x[categorical_columns].astype('float32')
test_x[categorical_columns]=test_x[categorical_columns].astype('float32')
from xgboost import XGBRegressor


xgb_params = {
    'learning_rate':0.2,
    'max_depth':5,
    'min_child_weight':300, 
   'colsample_bytree':0.8, 
    'subsample':0.8, 
    'eta':0.3,    
    'seed':42,
'tree_method':'gpu_hist'}

xgb_model = XGBRegressor(**xgb_params,n_estimators=3000)
xgb_model.fit(train_x, train_y, eval_set=[(val_x, val_y)], early_stopping_rounds=50, verbose=10)

xgb_preds = np.clip(xgb_model.predict(test_x),0,20)
print('RMSE on test: ', np.sqrt(mean_squared_error(test_y,xgb_preds)))
X = data[features]
X[categorical_columns] = X[categorical_columns].astype('int32')
Y=data[target]


from xgboost import XGBRegressor


xgb_params = {
    'learning_rate':0.2,
    'max_depth':5,
    'min_child_weight':300, 
   'colsample_bytree':0.8, 
    'subsample':0.8, 
    'eta':0.3,    
    'seed':42,
'tree_method':'gpu_hist'}

xgb_model_full = XGBRegressor(**xgb_params,n_estimators=3000)
xgb_model_full.fit(X,Y,verbose=500)
!pip install fastai2
!pip install fast_tabnet
from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *
train['date'] = pd.to_datetime(train['date'])
min_date = train.date.min()
max_date = train.date.max()
dates=pd.date_range(min_date,max_date,freq='MS').tolist()
dates = [pd.to_datetime(x) for x in dates]
blocks = range(0,len(dates))
dmy=pd.DataFrame({'date_block_num' : blocks,'ds' : dates})
fbdata=data[['date_block_num','item_sum_mth']].merge(dmy,
                                                     how='left',
                                                     on='date_block_num').rename(columns={'item_sum_mth' : 'y'})
from fbprophet import Prophet
prophet_model = Prophet()
prophet_model.fit(fbdata)
future = prophet_model.make_future_dataframe(periods=10, freq='MS')
fcast = prophet_model.predict(future)
fig = prophet_model.plot()
features
#submission[categorical_columns] = submission[categorical_columns].astype('str')
submission[categorical_columns] = submission[categorical_columns].astype('category')
#predictions
#cat_preds = np.clip(cat_model.predict(submission[features]),0,20)
#cat_model_full = cat_model_full.predict(submission[features])
#lgbm_preds = lgbm_model.predict(submission[features])


model = lgbm_model_full
preds = np.clip(model.predict(submission[features]),0,20)
submission['preds'] = preds
results = pd.DataFrame({'Id': submission.ID, 'item_cnt_month': submission.preds})
results.to_csv('submission.csv',index=False)
# Number of items available in test, but not in training

test_shops = test.shop_id.unique().tolist()

missing_items = list(np.setdiff1d(test.item_id.unique().tolist(),train.item_id.unique().tolist()))
missing_rows = test[test.item_id.isin(missing_items)].shape[0]
print('Number of items not available in training: ',len(missing_items))
print('Number of rows not available in training: ', missing_rows)
print('Number of rows for testing: ', test.shape[0])
print('Percentage missing: ', missing_rows * 100 / test.shape[0])

items_full=items.merge(item_categories,how='left',on=['item_category_id'])
missing=test[test.item_id.isin(missing_items)].merge(items_full,how='left',on='item_id')
agg_train = agg_train.merge(items[['item_id','item_category_id']],how='left',on='item_id')
# Generate a replacement item_id for items found in test, but missing in training
# Replacement strategy : most popular item within that category for that particular shop

item_cat_map = items[items.item_id.isin(missing_items)][['item_id','item_category_id']]
item_cat_map_dict = dict(zip(item_cat_map.item_id.values,item_cat_map.item_category_id.values))



most_pop_item_per_shop_and_cat=agg_train.groupby(['shop_id','item_category_id'])['item_id']\
                                               .apply(lambda x : x.value_counts().head(1))\
                                               .reset_index(name='occurences')\
                                               .rename(columns={'level_2' : 'item_id'})

most_pop_item_per_shop_and_cat['item_id'] = most_pop_item_per_shop_and_cat['item_id'].astype('int64')


# Assumption new items will sell similar to the most popular item within the same category sold at the same shop
replacement=missing.merge(most_pop_item_per_shop_and_cat,
                          how='left',
                          on=['shop_id','item_category_id'], 
                          suffixes=('_original','_replacement'))

# Replacement kind of work, but it seems that some shops are selling items in a category they havent done before
print(replacement[replacement.item_id_replacement.isna()].shape,replacement.shape)
print('\n')

train_shop_cat_map = agg_train.groupby(['shop_id'])['item_category_id'] \
                                     .apply(lambda x: x.unique().astype('int16').tolist())\
                                     .to_dict()
test_shop_cat_map = test.merge(items[['item_id','item_category_id']],how='left',on='item_id')\
                        .groupby(['shop_id'])['item_category_id']\
                        .apply(lambda x: x.unique().astype('int16').tolist())\
                        .to_dict()

# Check whats not matching up in train and test
result = {}
for shop, item_cat in test_shop_cat_map.items():
    #get item categories that are not in training but in test
    result[shop] = list(np.setdiff1d(item_cat,train_shop_cat_map[shop]))

import pprint 
pprint.pprint(result)
# Lets use historical data from other shops on scenarios where a shop has started selling a new item in a new category
replacement_no_nan=missing.merge(most_pop_item_per_shop_and_cat,
                                 how='left',
                                 on=['item_category_id'], 
                                 suffixes=('_original','_replacement'))
replacement.update(replacement_no_nan,overwrite=False)
replacement['item_id_replacement'] = replacement['item_id_replacement'] .astype('int16')
replacement['occurences'] = replacement['occurences'] .astype('int16')
# Generate the prediction dataframe for submission

cols = ['ID','shop_id','item_id_original','item_id_replacement']


# 1 add inn replacement item_ids
submission2 = test.merge(replacement[cols],
                        how='left',
                        left_on=['ID','shop_id','item_id'],
                        right_on=['ID','shop_id','item_id_original'],
                        suffixes=('','_repl'))
# 
submission2['item_id_new'] = np.where(submission2.item_id_replacement.isna(),
                                     submission2.item_id,
                                     submission2.item_id_replacement)

submission2['item_id_new'] = submission2['item_id_new'].astype('int16')

# 2 add in  year and month columns
last_month_w_sale=train[['shop_id','item_id','month_date']].drop_duplicates()\
                                                           .reset_index(drop=True)\
                                                           .sort_values(['shop_id','item_id','month_date'])\
                                                           .groupby(['shop_id','item_id'])\
                                                           .tail(1)
last_month_w_sale['add_1mth'] = last_month_w_sale.month_date + pd.DateOffset(months=1)
last_month_w_sale['last_year'] = pd.to_datetime(last_month_w_sale.month_date).dt.year
last_month_w_sale['last_month'] = pd.to_datetime(last_month_w_sale.month_date).dt.month
last_month_w_sale['year'] = pd.to_datetime(last_month_w_sale.add_1mth).dt.year
last_month_w_sale['month'] = pd.to_datetime(last_month_w_sale.add_1mth).dt.month
 

submission2 = submission2.merge(last_month_w_sale,how='left',
                              left_on=['shop_id','item_id_new'],
                              right_on=['shop_id','item_id'])

# Some items are completely new for the store, the last sold dates will then be drawn from
# the latest date _any_ item has been sold

# create a map to lookup the last sold date for any store
last_month_fillna=train[['shop_id','month_date']].drop_duplicates()\
                                                .reset_index(drop=True)\
                                                .sort_values(['shop_id','month_date'])\
                                                .groupby(['shop_id'])\
                                                .tail(1)

last_month_fillna['add_1mth'] = last_month_fillna.month_date + pd.DateOffset(months=1)
last_month_fillna['last_year'] = pd.to_datetime(last_month_fillna.month_date).dt.year
last_month_fillna['last_month'] = pd.to_datetime(last_month_fillna.month_date).dt.month
last_month_fillna['year'] = pd.to_datetime(last_month_fillna.add_1mth).dt.year
last_month_fillna['month'] = pd.to_datetime(last_month_fillna.add_1mth).dt.month

last_month_fillna_map=last_month_fillna[['shop_id','last_year','last_month','year','month']].groupby(['shop_id'])\
                                                                      .apply(lambda x : x.to_dict('records'))\
                                                                      .to_dict()

# Fill inn nans
submission2.last_year.fillna(submission2.shop_id.apply(lambda x : last_month_fillna_map[x][0]['last_year']),inplace=True)
submission2.last_month.fillna(submission2.shop_id.apply(lambda x : last_month_fillna_map[x][0]['last_month']),inplace=True)
submission2.year.fillna(submission2.shop_id.apply(lambda x : last_month_fillna_map[x][0]['year']),inplace=True)
submission2.month.fillna(submission2.shop_id.apply(lambda x : last_month_fillna_map[x][0]['month']),inplace=True)

# Fixed nans, downcast to int
cast_cols = ['last_year','last_month','year','month']
submission2[cast_cols] = submission2[cast_cols].astype('int16')

"""
# 3 add inn lagged features
submission2=submission2.merge(agg_train,
                how='left',
                left_on=['shop_id','item_id_new','last_year','last_month'],
                right_on=['shop_id','item_id','year','month'],
                suffixes=('','_lagged'))

# rename columns into lagged
rename_map = {'item_mean_mth':'item_mean_mth_lagged',
             'item_std_mth':'item_std_mth_lagged',
             'item_sum_mth':'item_sum_mth_lagged',
             'max_price':'max_price_lagged',
             'mean_price':'mean_price_lagged',
             'min_price':'min_price_lagged',
              'item_id' : 'item_id_old_2',
             'item_id_new' : 'item_id'}
submission2.rename(columns=rename_map,inplace=True)

# fix nans in item_category_id
item_cat_map = items[['item_id','item_category_id']].set_index(['item_id']).to_dict()['item_category_id']
submission.item_category_id.fillna(submission.item_id.apply(lambda x: item_cat_map[x]),inplace=True)
submission['item_category_id'] = submission['item_category_id'].astype('int16') 

# 4 select features
features = ['ID',
            'shop_id',
            'item_id',
            'item_category_id',
            'year',
             'month',
             'item_mean_mth_lagged',
             'item_std_mth_lagged',
             'item_sum_mth_lagged',
             'max_price_lagged',
             'mean_price_lagged',
             'min_price_lagged']

submission = submission[features]

# fill the rest of nans with 0, this should be okay as the lagged features entail information of last time this item
# was sold, however the remaining corner case is that the store has not sold this item before. Hence fillna with 0 
# should be ok
submission.fillna(0,inplace=True)
"""
