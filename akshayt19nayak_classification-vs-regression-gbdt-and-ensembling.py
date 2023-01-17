# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import gc

import warnings

from itertools import product

warnings.filterwarnings('ignore')

import os

import seaborn as sns

import lightgbm 

from xgboost import XGBRegressor, plot_importance

from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Lasso, Ridge

from sklearn.model_selection import KFold, cross_val_predict

from sklearn.externals import joblib

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
# Reading the files

root_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

df_test = pd.read_csv(root_path + 'test.csv')

df_item_categories = pd.read_csv(root_path + 'item_categories.csv')

df_sales_train = pd.read_csv(root_path + 'sales_train.csv')

df_sample_submission = pd.read_csv(root_path + 'sample_submission.csv')

df_items = pd.read_csv(root_path + 'items.csv')

df_shops = pd.read_csv(root_path + 'shops.csv')

df_categories_translated = pd.read_csv('/kaggle/input/translated/item_categories-translated.csv')

df_shops_translated = pd.read_csv('/kaggle/input/translated/shops-translated.csv')

print('The files have been loaded!')



print('df_test')

print(df_test.shape)

display(df_test.head(3))



print('df_item_categories')

print(df_item_categories.shape)

display(df_item_categories.head(3))



print('df_sales_train')

print(df_sales_train.shape)

display(df_sales_train.head(3))



print('df_sample_submission')

print(df_sample_submission.shape)

display(df_sample_submission.head(3))



print('df_items')

print(df_items.shape)

display(df_items.head(3))



print('df_shops')

print(df_shops.shape)

display(df_shops.head(3))
df_sales_timeseries = df_sales_train.groupby(['shop_id','item_id','date_block_num']).agg({'item_cnt_day':sum}).unstack(level=2)

df_sales_timeseries.columns = df_sales_train['date_block_num'].unique()

df_sales_timeseries = df_sales_timeseries.fillna(value=0)

plt.plot(df_sales_timeseries.sum(axis=0))
plt.plot(df_sales_timeseries.sum(axis=0).values[:12])

plt.plot(df_sales_timeseries.sum(axis=0).values[12:24])

plt.plot(df_sales_timeseries.sum(axis=0).values[24:])

plt.xticks(np.arange(12),np.arange(1,13))
print('The number of unique shop ids is', df_test['shop_id'].nunique())

print('The number of unique item ids is', df_test['item_id'].nunique())

print('The average number of item ids per shop is', df_test.groupby('shop_id').agg(count=('item_id', 'count')).mean()[0])
train_combos = df_sales_train.drop_duplicates(['shop_id','item_id']).shape[0]

test_combos = df_test.drop_duplicates(['shop_id','item_id']).shape[0]

seen_combos = pd.merge(df_sales_train.drop_duplicates(['shop_id','item_id']), df_test, on=['shop_id','item_id'], how='inner').shape[0]

print("The number of train combos is {} and the number of test combos is {}".format(train_combos, test_combos))

print("The number of combos in test that are present in train is {}".format(seen_combos))
df_test = pd.merge(df_test, df_items[['item_id','item_category_id']], on=['item_id'], how='left')

df_sales_train = pd.merge(df_sales_train, df_items[['item_id','item_category_id']], on=['item_id'], how='left')

unseen_item_categories = set(df_test['item_category_id']) - set(df_sales_train['item_category_id'])

print('The number of item categories that have never been sold before is', len(unseen_item_categories))
unseen_items = set(df_test['item_id']) - set(df_sales_train['item_id'])

print('The number of items that have never been sold before is', len(unseen_items))
df_item_price = df_sales_train.groupby('item_id').agg({'item_price':[np.mean, np.std,'count']})

# Since the NaN values are for those items that have a count of 1

df_item_price.fillna(0)

df_item_price.columns = ['mean','std','count']

print(df_item_price['std'].describe(percentiles=np.linspace(0,1,11)[1:10]))

plt.figure(figsize=(12,5))

plt.subplot(121)

sns.boxplot(df_item_price['std'])



plt.subplot(122)

sns.boxplot(df_item_price[df_item_price['std']>500]['mean'])
# Now if we were to look at variance of prices of the item category

df_item_category_price = df_sales_train.groupby('item_category_id').agg({'item_price':[np.mean, np.std]})

# Since the NaN values are for those items that have a count of 1

df_item_category_price.fillna(0)

df_item_category_price.columns = ['mean','std']

print(df_item_category_price['std'].describe(percentiles=np.linspace(0,1,11)[1:10]))

sns.boxplot(df_item_category_price['std'])
# To free up RAM

del df_sales_timeseries, unseen_items, unseen_item_categories, df_item_price, df_item_category_price

gc.collect()
# Deduplication. There are a few shop_ids that have the same names and similar shop names are mapped to the ids contained in the test set

df_sales_train.loc[df_sales_train.shop_id == 0, 'shop_id'] = 57

df_test.loc[df_test.shop_id == 0, 'shop_id'] = 57

df_sales_train.loc[df_sales_train.shop_id == 1, 'shop_id'] = 58

df_test.loc[df_test.shop_id == 1, 'shop_id'] = 58

df_sales_train.loc[df_sales_train.shop_id == 11, 'shop_id'] = 10

df_test.loc[df_test.shop_id == 11, 'shop_id'] = 10

df_sales_train.loc[df_sales_train.shop_id == 40, 'shop_id'] = 39

df_test.loc[df_test.shop_id == 40, 'shop_id'] = 39



# Extracting city code

df_shops_translated['shop_name'] = df_shops_translated['shop_name_translated'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()

df_shops_translated['shop_city'] = df_shops_translated['shop_name'].str.partition(' ')[0]

display(df_shops_translated[df_shops_translated['shop_id'].isin([0,57,1,58,11,10,40,39])])

df_shops_translated['city_code'] = LabelEncoder().fit_transform(df_shops_translated['shop_city'])

df_shops_translated = df_shops_translated[['shop_id','city_code']]



# Extracting type and subtype of categories

df_categories_translated['split'] = df_categories_translated['item_category_name_translated'].str.lower().str.split('-')

df_categories_translated['type'] = df_categories_translated['split'].map(lambda x: x[0].strip())

df_categories_translated['subtype'] = df_categories_translated['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

display(df_categories_translated.head())

df_categories_translated['type_code'] = LabelEncoder().fit_transform(df_categories_translated['type'])

df_categories_translated['subtype_code'] = LabelEncoder().fit_transform(df_categories_translated['subtype'])

df_categories_translated = df_categories_translated[['item_category_id','type_code','subtype_code']]
df_sales_grouped = df_sales_train.groupby(['shop_id','item_id','date_block_num']).agg(item_cnt_month=('item_cnt_day',sum)).reset_index()

df_prices_grouped = df_sales_train.groupby(['date_block_num','item_id']).agg(item_price=('item_price','mean')).reset_index() # aggregating price for each month using mean

df_prices_grouped['item_price_bin'], bins =  pd.qcut(df_prices_grouped['item_price'], 7, labels=np.arange(1,8), retbins=True) # binning prices

df_test['date_block_num'] = 34

df_test['item_cnt_month'] = np.nan



# For every month we create a grid from all shops/items combinations from that month

grid = [] 

grid_cols = ['shop_id', 'item_id', 'date_block_num']

for date_block_num in df_sales_train['date_block_num'].unique():

    cur_shops = df_sales_train.loc[df_sales_train['date_block_num'] == date_block_num, 'shop_id'].unique()

    cur_items = df_sales_train.loc[df_sales_train['date_block_num'] == date_block_num, 'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [date_block_num]])),dtype='int32'))

df_grid = pd.DataFrame(np.vstack(grid), columns = grid_cols, dtype=np.int32)

del grid

df_grid = pd.merge(df_grid, df_sales_grouped, on=['shop_id','item_id','date_block_num'], how='left') # to get other attributes from the sales data

df_grid['item_cnt_month'] = df_grid['item_cnt_month'].fillna(0).clip(0,20) # clipping values so that the train data is representative of the test data

df_grid = pd.concat([df_grid, df_test.drop(['item_category_id','ID'], axis=1)], axis=0, ignore_index=True) # concatenating to preprocess together

df_grid = pd.merge(df_grid, df_items[['item_id','item_category_id']], on='item_id', how='left') # to get item_category_id

df_grid = pd.merge(df_grid, df_shops_translated, on='shop_id', how='left') # to get city_code

df_grid = pd.merge(df_grid, df_categories_translated, on='item_category_id', how='left') # to get type_code and subtype_code

df_grid = pd.merge(df_grid, df_prices_grouped, on=['date_block_num','item_id'], how='left') # to get prices

print('The grid has been created!')

del df_sales_grouped, df_prices_grouped

gc.collect()
# Imputing prices for the test set (ugly code). The logic is defined in the EDA section

df_prices_shop_item = df_sales_train[['shop_id','item_id','item_price']].drop_duplicates(subset=['shop_id','item_id'], keep='last') 

df_prices_item = df_sales_train[['item_id','item_price']].drop_duplicates(subset=['item_id'], keep='last')

df_prices_category = df_sales_train.groupby('item_category_id').agg(item_price=('item_price','mean')).reset_index()



df_test = df_grid[df_grid['date_block_num']==34]

df_test = pd.merge(df_test.drop('item_price',axis=1), df_prices_shop_item, on=['shop_id','item_id'], how='left')

item_price = pd.merge(df_test[pd.isnull(df_test['item_price'])], df_prices_item, on='item_id', 

                      how='left').set_index(df_test[pd.isnull(df_test['item_price'])].index)['item_price_y']

df_test['item_price'] = df_test['item_price'].fillna(item_price)

category_price = pd.merge(df_test[pd.isnull(df_test['item_price'])], df_prices_category, on='item_category_id', 

                          how='left').set_index(df_test[pd.isnull(df_test['item_price'])].index)['item_price_y']

df_test['item_price'] = df_test['item_price'].fillna(category_price)

df_test['item_price_bin'] = pd.cut(df_test['item_price'], bins=bins, labels=np.arange(1,8), include_lowest=True)



# Concatenating

df_grid = pd.concat([df_grid[df_grid['date_block_num']<34], df_test[df_grid.columns]], axis=0)

df_grid['item_price_bin'] = df_grid['item_price_bin'].astype(int)

del df_prices_shop_item, df_prices_item, df_prices_category, df_test, df_sales_train

gc.collect()
# Feature engineering

# The first few variables are group means of date_block_num + additional categorical variables

df_grid['date_item_mean'] = df_grid.groupby(['date_block_num','item_id'])['item_cnt_month'].transform('mean') 

df_grid['date_category_mean'] = df_grid.groupby(['date_block_num','item_category_id'])['item_cnt_month'].transform('mean')

df_grid['date_item_shop_mean'] = df_grid.groupby(['date_block_num','item_id','shop_id'])['item_cnt_month'].transform('mean')

df_grid['date_item_category_mean'] = df_grid.groupby(['date_block_num','item_id','item_category_id'])['item_cnt_month'].transform('mean')

df_grid['date_shop_category_mean'] = df_grid.groupby(['date_block_num','shop_id','item_category_id'])['item_cnt_month'].transform('mean')

df_grid['month'] = df_grid['date_block_num'] % 12

number_of_days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

df_grid['days'] = df_grid['month'].map(number_of_days)

# Time dependent information

df_grid['months_since_item_sale'] = df_grid['date_block_num'] - df_grid.groupby('item_id')['date_block_num'].transform('min')

df_grid['months_since_item_shop_sale'] = df_grid['date_block_num'] - df_grid.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
# Shop revenue

df_grid['shop_item_revenue'] = df_grid['item_cnt_month']*df_grid['item_price']

df_revenue = df_grid.groupby(['shop_id','date_block_num']).agg(shop_revenue=('shop_item_revenue',sum)).reset_index()

df_grid = pd.merge(df_grid, df_revenue, on=['shop_id','date_block_num'], how='left')

df_grid.drop('shop_item_revenue', axis=1, inplace=True) 

del df_revenue

gc.collect()
# Downcasting data types to save memory

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

    df[int_cols]   = df[int_cols].astype(np.int16)    

    return df

shop_revenue_cols = [col for col in df_grid.columns if 'shop_revenue' in col]

cols_to_downcast = list(set(df_grid.columns) - set(shop_revenue_cols))

df_grid[shop_revenue_cols] = df_grid[shop_revenue_cols].astype(np.float32) 

df_grid[cols_to_downcast] = downcast_dtypes(df_grid[cols_to_downcast])
# Lagging 

def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    df.drop(col, axis=1, inplace=True)

    del shifted

    return(df)



mean_encoded = [col for col in df_grid.columns if 'mean' in col]

cols_to_lag = ['shop_revenue','item_price'] + mean_encoded

for col in cols_to_lag:

    df_grid = lag_feature(df_grid, [1,2,3,6,12], col)

    print('Lagged values for {} have been calculated'.format(col))

df_grid = df_grid[df_grid['date_block_num']>=12] # Since we are using a lag of 12, the feature set for the first 12 month is not useful

lagged_cols = [col for col in df_grid.columns if 'lag' in col]

df_grid[lagged_cols] = df_grid[lagged_cols].fillna(0)

df_grid.reset_index(drop=True, inplace=True)

gc.collect()
# And this is how it looks 

df_grid.head()
# Run this cell if the kernel gets killed in between. It contains the feature set for the models built below

df_grid = pd.read_feather('/kaggle/input/data-prep/df_sales.feather')

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

    df[float_cols] = df[float_cols].astype(np.float16)

    df[int_cols]   = df[int_cols].astype(np.int16)    

    return df

shop_revenue_cols = [col for col in df_grid.columns if 'shop_revenue' in col]

cols_to_downcast = list(set(df_grid.columns) - set(shop_revenue_cols))

df_grid[shop_revenue_cols] = df_grid[shop_revenue_cols].astype(np.float32) # To prevent numerical overflow

df_grid[cols_to_downcast] = downcast_dtypes(df_grid[cols_to_downcast])
# Dividing into train, validation and test. We use months upto 33 for training, the 33rd month for validation and the last (34th month) for testing

train = df_grid[df_grid['date_block_num']<33]

valid = df_grid[df_grid['date_block_num']==33]

X_test = df_grid[df_grid['date_block_num']==34].drop('item_cnt_month', axis=1)



# Dividing into feature and label for regressor

X_train = train[train['item_cnt_month']>0].drop('item_cnt_month', axis=1)

y_train = train[train['item_cnt_month']>0]['item_cnt_month']

X_valid = valid[valid['item_cnt_month']>0].drop('item_cnt_month', axis=1)

y_valid = valid[valid['item_cnt_month']>0]['item_cnt_month']



# For storing OOF predictions

df_classreg_valid  = pd.DataFrame()

df_classreg_test = pd.DataFrame()

df_reg_valid  = pd.DataFrame()

df_reg_test = pd.DataFrame()



del df_grid

gc.collect()
# LightGBM regressor

regressor_params = {

               'feature_fraction': 0.75,

               'metric': 'rmse',

               'nthread': 1, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.75, 

               'learning_rate': 0.03, 

               'objective': 'mse', 

               'bagging_seed': 2**7, 

               'num_leaves': 80,

               'bagging_freq':1,

               'verbose':0 

              }

lightgbm_train = lightgbm.Dataset(X_train, label=y_train)

lightgbm_valid = lightgbm.Dataset(X_valid, label=y_valid)

regressor = lightgbm.train(regressor_params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_valid], valid_names=['train','valid'], verbose_eval=10, num_boost_round=500, early_stopping_rounds=10)

print('LightGBM model for regression has been built!')

df_classreg_valid['lightgbm'] = regressor.predict(valid.drop('item_cnt_month', axis=1))

df_classreg_test['lightgbm'] = regressor.predict(X_test)

lightgbm.plot_importance(regressor, figsize=(10,12))

regressor.save_model('lightgbm_classreg_regressor.txt', num_iteration=regressor.best_iteration) # Load it back using, model = lightgbm.Booster(model_file='lightgbm_classreg_regressor.txt')

del lightgbm_train, lightgbm_valid, regressor

gc.collect()
# XGBoost regressor

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



regressor = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=100, 

    colsample_bytree=0.75, 

    subsample=0.75, 

    eta=0.1,    

    seed=42)



regressor.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, y_train), (X_valid, y_valid)], 

    verbose=10, 

    early_stopping_rounds = 10)

print('XGBoost model for regression has been built!')

df_classreg_valid['xgboost'] = regressor.predict(valid.drop('item_cnt_month', axis=1))

df_classreg_test['xgboost'] = regressor.predict(X_test)

plot_features(regressor, (10,12))

regressor.save_model('xgboost_classreg_regressor.model') # Load it back using, model = xgboost.Booster(), model.load_model('xgboost_classreg_regressor.model')

del regressor

gc.collect()
# CatBoost regressor

cat_features = ['shop_id','item_id','item_category_id','item_price_bin','month','date_block_num','city_code','type_code','subtype_code']

regressor = CatBoostRegressor(

    iterations=100,

    random_seed=0,

    learning_rate=0.1,

    max_ctr_complexity=3, # To enable feature interactions

    has_time=True, # To disable random permutations

    boosting_type='Ordered', # To reduce overfitting

    loss_function='RMSE',

    od_type='Iter', 

    od_wait=10, # Early stopping

)

regressor.fit(

    X_train, y_train,

    cat_features=cat_features,

    eval_set=(X_valid, y_valid),

    verbose=10

)

print('CatBoost model for regression has been built!')

df_classreg_valid['catboost'] = regressor.predict(valid.drop(['item_cnt_month'], axis=1))

df_classreg_test['catboost'] = regressor.predict(X_test)

print(regressor.get_feature_importance(prettified=True))

regressor.save_model('catboost_classreg_regressor.bin') # Load it back using model = CatBoostRegressor(), model.load_model('catboost_classreg_regressor.bin')

del regressor

gc.collect()
# LightGBM classifier

classifier_params = {

               'feature_fraction': 0.75,

               'metric': 'auc', 

               'nthread': 1, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.75, 

               'learning_rate': 0.3, 

               'objective': 'binary', 

               'bagging_seed': 2**7, 

               'num_leaves': 80,

               'bagging_freq':1,

               'is_unbalance':True,

               'verbose':0 

              }

lightgbm_train = lightgbm.Dataset(train.drop('item_cnt_month', axis=1), label=train['item_cnt_month'].apply(lambda x: 1 if x >= 1 else 0))

lightgbm_valid = lightgbm.Dataset(valid.drop('item_cnt_month', axis=1), label=valid['item_cnt_month'].apply(lambda x: 1 if x >= 1 else 0))

classifier = lightgbm.train(classifier_params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_valid], valid_names=['train','valid'], verbose_eval=10, num_boost_round=300, early_stopping_rounds=10)

print('Model for classification has been built!')

df_classreg_valid['prob'] = classifier.predict(valid.drop('item_cnt_month', axis=1))

df_classreg_test['prob'] = classifier.predict(X_test)

classifier.save_model('lightgbm_classreg_classifier.txt', num_iteration=classifier.best_iteration) # Can load it back using model = lightgbm.Booster(model_file='lightgbm_classreg_classifier.txt')

del lightgbm_train, lightgbm_valid, classifier

gc.collect()
df_classreg_valid['average'] = (df_classreg_valid['lightgbm'] + df_classreg_valid['xgboost'] + df_classreg_valid['catboost'])/3

df_classreg_valid['item_cnt_month'] = df_classreg_valid.apply(lambda row: row['average'] if row['prob']>=0.8 else 0, axis=1) 

print('Validation RMSE is', np.sqrt(mean_squared_error(valid['item_cnt_month'], df_classreg_valid['item_cnt_month'])))

df_classreg_valid.to_csv('classreg_valid.csv', index=False)



df_classreg_test['average'] = (df_classreg_test['lightgbm'] + df_classreg_test['xgboost']+ df_classreg_test['catboost'])/3

df_classreg_test['item_cnt_month'] = df_classreg_test.apply(lambda row: row['average'] if row['prob']>=0.8 else 0, axis=1) 

df_classreg_test['ID'] = df_sample_submission['ID']

df_classreg_test = df_classreg_test[['ID','item_cnt_month']]

df_classreg_test.to_csv('classreg_test.csv', index=False) # Final submission file
# Part of this code has been borrowed from this fantastic kernel https://www.kaggle.com/mbrown89/boost-your-score-guaranteed-leaderboard-probing

path = '/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv'

train_pairs = pd.read_csv(path)

train_pairs = train_pairs[train_pairs['date_block_num']<33][['shop_id','item_id']].drop_duplicates(['shop_id','item_id'])

pairs={(a, b) for a, b in zip(train_pairs.shop_id, train_pairs.item_id)}

items={a for a in train_pairs.item_id}

df_reg_valid['group'] = [2 if (a,b) in pairs else (1 if b in items else 0) for a,b in zip(valid.shop_id, valid.item_id)]

df_reg_test['group'] = [2 if (a,b) in pairs else (1 if b in items else 0) for a,b in zip(X_test.shop_id, X_test.item_id)]

del train_pairs, pairs, items

gc.collect()
# LightGBM model fitting

lightgbm_params = {

               'feature_fraction': 0.75,

               'metric': 'rmse',

               'nthread':1, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.75, 

               'learning_rate': 0.03, 

               'objective': 'mse', 

               'bagging_seed': 2**7, 

               'num_leaves': 80,

               'bagging_freq':1,

               'verbose':0 

              }

lightgbm_train = lightgbm.Dataset(train.drop('item_cnt_month', axis=1), label=train['item_cnt_month'])

lightgbm_valid = lightgbm.Dataset(valid.drop('item_cnt_month', axis=1), label=valid['item_cnt_month'])

model = lightgbm.train(lightgbm_params, lightgbm_train, valid_sets=[lightgbm_train, lightgbm_valid], valid_names=['train','valid'], verbose_eval=10, num_boost_round=300, 

                       early_stopping_rounds=10)

df_reg_valid['lightgbm'] = model.predict(valid.drop('item_cnt_month', axis=1))

df_reg_test['lightgbm'] = model.predict(X_test)

model.save_model('lightgbm_reg.txt', num_iteration=model.best_iteration) # Load it back using, model = lightgbm.Booster(model_file='lightgbm_reg.txt')

del lightgbm_train, lightgbm_valid, model

gc.collect()
# Dividing into train, validation and test for ridge regression

X_train_iloc = train.index[-1] + 1

X_valid_iloc = valid.index[-1] + 1

y_train =  train['item_cnt_month']

y_valid = valid['item_cnt_month']

del train, valid

gc.collect()
# Scaling data

df_scaled = pd.read_feather('/kaggle/input/scaling/df_scaled.feather') # Kernel runs out of memory, hence the data has been preprocessed elsewhere

df_scaled = downcast_dtypes(df_scaled)

X_train = df_scaled.iloc[:X_train_iloc]

X_valid = df_scaled.iloc[X_train_iloc:X_valid_iloc]

X_test = df_scaled.iloc[X_valid_iloc:]

del df_scaled

gc.collect()

print('The scaled variables are', X_train.columns)
def plot_coef(coef, predictors):

    plt.figure(figsize=(10,8))

    coef = pd.Series(coef, predictors).sort_values()

    coef.plot(kind='bar', title='Model Coefficients')

ridge = Ridge(alpha=10)

ridge.fit(X_train.values, y_train.values)

joblib.dump(ridge, 'ridge_reg.pkl') # Load it back using, model = joblib.load('ridge_reg.pkl') 

print('The RMSE for ridge regression is', np.sqrt(mean_squared_error(y_valid, ridge.predict(X_valid))))

plot_coef(ridge.coef_, X_train.columns)

df_reg_valid['ridge'] = ridge.predict(X_valid)

df_reg_test['ridge'] = ridge.predict(X_test)
lasso = Lasso(alpha=0.05)

lasso.fit(X_train.values, y_train.values)

print('The RMSE for lasso regression is', np.sqrt(mean_squared_error(y_valid, lasso.predict(X_valid))))

plot_coef(lasso.coef_, X_train.columns)
#Shallow depth

ensembler = XGBRegressor(

    max_depth=2,

    n_estimators=150,

    min_child_weight=100, 

    colsample_bytree=0.75, 

    subsample=0.75, 

    eta=0.1,    

    seed=42)

kfold = KFold(n_splits=5, random_state=42)

y_pred = cross_val_predict(ensembler, df_reg_valid, y_valid, cv=kfold)

print('The cross validation RMSE is', np.sqrt(mean_squared_error(y_valid, y_pred)))
# To inspect predicted means and target means for the individual models and the ensemble

df_mean_analysis = pd.DataFrame()

df_mean_analysis['group'] = df_reg_valid['group']

df_mean_analysis['lightgbm'] = df_reg_valid['lightgbm']

df_mean_analysis['ridge'] = df_reg_valid['ridge']

df_mean_analysis['ensemble'] = y_pred

df_mean_analysis['target'] = y_valid.values

df_mean_analysis.groupby('group').agg(np.mean)
ensembler.fit(df_reg_valid, y_valid)

ensembler.save_model('ensembler.model') # Load it back using, model = xgboost.Booster(), model.load_model('ensembler.model')

plot_importance(ensembler)
# Same analysis for the test set

df_mean_analysis = pd.DataFrame()

df_mean_analysis['group'] = df_reg_test['group']

df_mean_analysis['lightgbm'] = df_reg_test['lightgbm']

df_mean_analysis['ridge'] = df_reg_test['ridge']

df_mean_analysis['pred'] = ensembler.predict(df_reg_test)

df_mean_analysis.groupby('group').agg(np.mean)
del df_mean_analysis

gc.collect()
df_reg_valid.to_csv('reg_valid.csv', index=False)

df_reg_test.to_csv('reg_preds_test.csv', index=False)

df_reg_test['item_cnt_month'] = ensembler.predict(df_reg_test)

df_reg_test['ID'] = df_sample_submission['ID']

df_reg_test = df_reg_test[['ID','item_cnt_month']]

df_reg_test.to_csv('reg_test.csv', index=False) # Final submission file