import gc
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold

import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# Useful functions
def submit(y_pred, fname='submission'):
    test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
    submission = pd.DataFrame({
        'ID': test['ID'],
        'item_cnt_month': y_pred.ravel()
    })
    submission.to_csv(f'{fname}.csv', index=False)
    
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

def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def plot_seasonal_decompose(result, title=''):
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(result.observed)
    plt.ylabel('Observed')
    
    plt.subplot(412)
    plt.plot(result.trend)
    plt.ylabel('Trend')
    
    plt.subplot(413)
    plt.plot(result.seasonal)
    plt.ylabel('Seasonal')
    
    plt.subplot(414)
    plt.plot(result.resid)
    plt.ylabel('Residual')
    plt.suptitle(title)
    plt.show()
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
items2 = pd.read_csv('../input/predict-future-sales-russian-translated/items-translated.csv')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories2 = pd.read_csv('../input/predict-future-sales-russian-translated/item_categories-classified.csv')

item_categories_classes = pd.read_csv('../input/predict-future-sales-russian-translated/item_categories-classes.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
shops2 = pd.read_csv('../input/predict-future-sales-russian-translated/shops-translated.csv')

sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
print('items :', items.shape)
print('item_categories :', item_categories.shape)
print('shops :', shops.shape)
print('sales :', sales.shape)
print('test :', test.shape)
items.head()
item_categories.head()
shops.head()
sales.head()
test.head()
data = sales.groupby('date_block_num')['item_cnt_day'].sum()

plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(data)
plt.xlabel('month')
plt.ylabel('items sold')

# Rolling statistics
plt.subplot(122)
plt.plot(data.rolling(window=12).mean(), label='rolling mean')
plt.plot(data.rolling(window=12).std(), label='rolling std')
plt.xlabel('month')
plt.ylabel('items sold')
plt.legend()
plt.show()
result = seasonal_decompose(data.values, freq=12, model='additive')
plot_seasonal_decompose(result, title='additive')
result = seasonal_decompose(data.values, freq=12, model='multiplicative')
plot_seasonal_decompose(result, title='multiplicative')
del data, result
data = items.groupby('item_category_id')['item_id'].count().sort_values(ascending=False).reset_index(name='item_count')

plt.figure(figsize=(10, 5))
sns.barplot(x='item_category_id', y='item_count', data=data[:10])
plt.show()
del data
top_items = sales.groupby('item_id').sum()['item_cnt_day'].sort_values(ascending=False)
top_items = pd.merge(top_items, items2, how='left', on='item_id')

sns.barplot(y='item_name_translated', x='item_cnt_day', data=top_items[:10])
del top_items
top_categories = pd.merge(sales, items, how='left', on='item_id').groupby('item_category_id').sum()['item_cnt_day'].sort_values(ascending=False)
top_categories = pd.merge(top_categories, item_categories2, how='left', on='item_category_id')

sns.barplot(y='item_category_name_translated', x='item_cnt_day', data=top_categories[:10])
del top_categories
top_category_classes = pd.merge(sales, items, how='left', on='item_id').groupby('item_category_id').sum()['item_cnt_day'].sort_values(ascending=False)
top_category_classes = pd.merge(top_category_classes, pd.merge(item_categories2, item_categories_classes, how='left', on='item_class_id'), how='left', on='item_category_id')
top_category_classes = top_category_classes.groupby('item_class_name').sum().sort_values('item_cnt_day', ascending=False).reset_index()

sns.barplot(y='item_class_name', x='item_cnt_day', data=top_category_classes)
del top_category_classes
top_shops = sales.groupby('shop_id').sum()['item_cnt_day'].sort_values(ascending=False)
top_shops = pd.merge(top_shops, shops2, how='left', on='shop_id')

sns.barplot(y='shop_name_translated', x='item_cnt_day', data=top_shops[:10])
del top_shops
top_shops = sales.groupby('shop_id').sum()['item_cnt_day']
top_shops = pd.merge(top_shops, shops2, how='left', on='shop_id')
top_shop_locations = top_shops.groupby('shop_location').sum().sort_values('item_cnt_day', ascending=False).reset_index()

sns.barplot(y='shop_location', x='item_cnt_day', data=top_shop_locations[:10])
del top_shops, top_shop_locations
plt.figure(figsize=(10, 4))
sns.boxplot(sales['item_cnt_day'])
plt.show()
plt.figure(figsize=(10, 4))
sns.boxplot(sales['item_price'])
plt.show()
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in tqdm_notebook(sales['date_block_num'].unique()):
    cur_shops = sales[sales['date_block_num'] == block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num'] == block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))

# turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

# Groupby data to get shop-item-month aggregates
gb = sales.groupby(index_cols, as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})
# fix column names
gb.rename(columns={'item_cnt_day': 'target', 'item_price': 'price_target'}, inplace=True)
# Join it to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

# Same as above but with shop-month aggregates
gb = sales.groupby(['shop_id', 'date_block_num'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})
gb.rename(columns={'item_cnt_day': 'target_shop', 'item_price': 'price_target_shop'}, inplace=True)
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

# Same as above but with item-month aggregates
gb = sales.groupby(['item_id', 'date_block_num'], as_index=False).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})
gb.rename(columns={'item_cnt_day': 'target_item', 'item_price': 'price_target_item'}, inplace=True)
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)
del grid, gb
gc.collect();

all_data.head()
test_ = test.drop(columns=['ID'])
test_['date_block_num'] = 34
all_data = pd.concat([all_data, test_])
del test_
cols_to_rename = list(all_data.columns.difference(index_cols))

shift_range = [1, 2, 3, 4, 5, 6, 11, 12]

for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    train_shift = train_shift.rename(columns=lambda c: f'{c}_lag_{month_shift}' if c in cols_to_rename else c)
    
    all_data = all_data.merge(train_shift, how='left', on=index_cols).fillna(0)
    
del train_shift
lag_features = [c for c in all_data.columns if c[-1] in [str(x) for x in shift_range]]

# Don't use old data from year 2013
all_data = all_data[all_data['date_block_num'] >= 12] 
# 1
all_data = pd.merge(all_data, items[['item_id', 'item_category_id']], how='left', on='item_id')
# 2
all_data = pd.merge(all_data, item_categories2[['item_category_id', 'item_class_id']], how='left', on='item_category_id')
# 3
shops2['shop_location_id'] = pd.factorize(shops2['shop_location'])[0]
all_data = pd.merge(all_data, shops2[['shop_id', 'shop_location_id']], how='left', on='shop_id')

all_data = downcast_dtypes(all_data)
gc.collect();
print(all_data.shape)
all_data.head()
def mean_enc_cv(df, col, target, n_splits=5):
    df[f'{col}_{target}_mean_enc'] = np.nan
    
    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_index, val_index in tqdm_notebook(kf.split(df)):
        X_train, X_val = df.iloc[train_index], df.iloc[val_index]
        means = X_train.groupby(col)[target].mean()
        X_val[f'{col}_{target}_mean_enc'] = X_val[col].map(means)
        df.iloc[val_index] = X_val
        
    # Fill NaNs
    df[f'{col}_{target}_mean_enc'] = df[f'{col}_{target}_mean_enc'].fillna(df[target].mean())
mean_enc_cv(all_data, 'item_id', 'target')
mean_enc_cv(all_data, 'item_id', 'price_target')
mean_enc_cv(all_data, 'item_id', 'target_shop')
mean_enc_cv(all_data, 'shop_id', 'price_target')
mean_enc_cv(all_data, 'item_category_id', 'target')
mean_enc_cv(all_data, 'item_category_id', 'price_target')
mean_enc_cv(all_data, 'item_id', 'price_target_item')
mean_enc_cv(all_data, 'item_class_id', 'target')
mean_enc_cv(all_data, 'item_class_id', 'target_shop')
tfidf_features = 25

vectorizer = TfidfVectorizer(max_features=tfidf_features)
vectors = vectorizer.fit_transform(items['item_name'])

items_tfidf = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names())
items_tfidf = items_tfidf.rename(columns=lambda c: 'item_name_tfidf_' + c)
items_tfidf['item_id'] = items['item_id']

all_data = pd.merge(all_data, items_tfidf, how='left', on='item_id')
all_data = downcast_dtypes(all_data)
print(all_data.shape)
all_data.head()
gc.collect();
dates = all_data['date_block_num']

# Drop all target cols
to_drop_cols = ['target_shop', 'target_item', 'price_target', 'price_target_shop', 'price_target_item']
traintest = all_data.drop(columns=to_drop_cols)
tfidf_features = [c for c in all_data.columns if c.find('tfidf') != -1]

# Renaming 'traintest' for "not support non-ASCII characters" issue
rename_tfidf = {}
for i, col in enumerate(tfidf_features):
    new_col = ''
    for s in col.split('_')[:-1]:
        new_col += s + '_'
    new_col += str(i)
    rename_tfidf[col] = new_col

traintest.rename(columns=rename_tfidf, inplace=True)
tfidf_features = list(rename_tfidf.values())
del rename_tfidf
selector = VarianceThreshold(threshold=0.0)
selector.fit(traintest[dates <= 33])
traintest.columns[~selector.get_support()]
traintest.columns[traintest.nunique() == 1]
index_cols = ['shop_id', 'item_id', 'date_block_num']

features_not_used = index_cols + ['item_category_id',
                                 'item_class_id',
                                 'shop_location_id']

# Top tfidf features
tfidf_features = ['item_name_tfidf_24',
                 'item_name_tfidf_12',
                 'item_name_tfidf_21',
                 'item_name_tfidf_16',
                 'item_name_tfidf_7']
# Use top 30 features
features = ['target_item_lag_1', 'item_id_target_mean_enc',
           'shop_id_price_target_mean_enc', 'target_lag_1',
           'item_id_target_shop_mean_enc',
           'item_category_id_price_target_mean_enc',
           'item_class_id_target_mean_enc', 'item_category_id_target_mean_enc',
           'item_class_id_target_shop_mean_enc', 'item_id_price_target_mean_enc',
           'item_id_price_target_item_mean_enc', 'price_target_item_lag_1',
           'target_lag_2', 'target_item_lag_2', 'target_shop_lag_2',
           'target_item_lag_3', 'price_target_item_lag_2', 'price_target_lag_1',
           'target_item_lag_4', 'price_target_shop_lag_1', 'target_lag_3',
           'target_shop_lag_12', 'price_target_shop_lag_4', 'target_lag_5',
           'price_target_item_lag_3', 'target_lag_6', 'price_target_shop_lag_2',
           'target_item_lag_6', 'target_shop_lag_1', 'target_lag_4'] + ['target']

features += tfidf_features

features = set(features).difference(features_not_used)

traintest = traintest[features]
traintest.shape
train = traintest[dates <= 33]
X_test = traintest[dates == 34].drop(columns=['target'])

print(train.shape)
print(X_test.shape)
X = train.drop(columns=['target'])
y = train['target']

print(X.shape, y.shape)
del train, traintest
gc.collect();
def cv_time(X, y, model, date_block_nums):
    train_scores = []
    val_scores = []
    
    for date_block_num in tqdm_notebook(date_block_nums):
        X_train, y_train = X[dates < date_block_num], y[dates < date_block_num]
        X_val, y_val = X[dates == date_block_num], y[dates == date_block_num]
        
        y_train = y_train.clip(0, 20)
        lb, ub = np.percentile(y_val, (0.05, 99.9))
        y_val = y_val.clip(lb, ub)
        
        model.fit(X_train, y_train)
        
        train_scores.append(rmse(y_train, model.predict(X_train) ))
        val_scores.append(rmse(y_val, model.predict(X_val) ))
    
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    print('Train scores :', train_scores)
    print('Mean :', train_scores.mean())
    print()
    print('Val scores :', val_scores)
    print('Mean :', val_scores.mean())
cv_time(X, y, model=lgb.LGBMRegressor(), date_block_nums=[22, 33])
model_lgb = lgb.LGBMRegressor(n_estimators=200, 
                              reg_alpha=0.01)
model_lgb.fit(X, y.clip(0, 20))
y_pred = model_lgb.predict(X_test)
submit(y_pred, fname='submission-lgb-1')