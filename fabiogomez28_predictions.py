import itertools
from time import sleep
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
sns.set(rc={'figure.figsize':(20, 10)})
import xgboost as xgb


item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

groupby_cols = ['date_block_num', 'shop_id', 'item_id']
sns.set_context("talk", font_scale=1.4)
sales_month = pd.DataFrame(train.groupby(['date_block_num']).sum().item_cnt_day).reset_index()
sales_month.columns = ['date_block_num', 'sum_items_sold']
sns.barplot(x ='date_block_num', y='sum_items_sold', 
            data=sales_month.reset_index());
plt.plot(sales_month.sum_items_sold)
plt.title('Distribution of the sum of sales per month')
del sales_month

comb_shop_item = pd.DataFrame(train[['date_block_num', 'shop_id', 
                                     'item_id']].drop_duplicates().groupby('date_block_num').size()).reset_index()
comb_shop_item.columns = ['date_block_num', 'item-shop_comb']
sns.barplot(x ='date_block_num', y='item-shop_comb', data=comb_shop_item);
plt.plot(comb_shop_item['item-shop_comb']);
plt.title('Number of combinations shop-it with sales per month')
del comb_shop_item
sns.set_context("talk", font_scale=1)
sales_month_shop_id = pd.DataFrame(train.groupby(['shop_id']).sum().item_cnt_day).reset_index()
sales_month_shop_id.columns = ['shop_id', 'sum_sales']
sns.barplot(x ='shop_id', y='sum_sales', data=sales_month_shop_id, palette='Paired')
plt.title('Distribution of sales per shop');
del sales_month_shop_id
sns.set_context("talk", font_scale=1.4)
sales_item_id = pd.DataFrame(train.groupby(['item_id']).sum().item_cnt_day)
plt.xlabel('item id')
plt.ylabel('sales')
plt.plot(sales_item_id);
sns.set_context("talk", font_scale=0.8)
sales_item_cat = train.merge(items, how='left', on='item_id').groupby('item_category_id').item_cnt_day.sum()
sns.barplot(x ='item_category_id', y='item_cnt_day',
            data=sales_item_cat.reset_index(), 
            palette='Paired'
           );
del sales_item_cat
train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1001]

median = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (
            train.item_price > 0)].item_price.median()
train.loc[train.item_price < 0, 'item_price'] = median
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
test['date_block_num'] = 34
category = items[['item_id', 'item_category_id']].drop_duplicates()
category.set_index(['item_id'], inplace=True)
category = category.item_category_id
train['category'] = train.item_id.map(category)
item_categories['meta_category'] = item_categories.item_category_name.apply(lambda x: x.split(' ')[0])
item_categories['meta_category'] = pd.Categorical(item_categories.meta_category).codes
item_categories.set_index(['item_category_id'], inplace=True)
meta_category = item_categories.meta_category
train['meta_category'] = train.category.map(meta_category)
shops['city'] = shops.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
shops['city'] = pd.Categorical(shops['city']).codes
city = shops.city
train['city'] = train.shop_id.map(city)
year = pd.concat([train.date_block_num, train.date.apply(lambda x: int(x.split('.')[2]))], axis=1).drop_duplicates()
year.set_index(['date_block_num'], inplace=True)
year = year.date.append(pd.Series([2015], index=[34]))
month = pd.concat([train.date_block_num, train.date.apply(lambda x: int(x.split('.')[1]))], axis=1).drop_duplicates()
month.set_index(['date_block_num'], inplace=True)
month = month.date.append(pd.Series([11], index=[34]))
all_shops_items = []

for block_num in train['date_block_num'].unique():
    unique_shops = train[train['date_block_num'] == block_num]['shop_id'].unique()
    unique_items = train[train['date_block_num'] == block_num]['item_id'].unique()
    all_shops_items.append(np.array(list(itertools.product([block_num], unique_shops, unique_items)), dtype='int32'))

df = pd.DataFrame(np.vstack(all_shops_items), columns=groupby_cols, dtype='int32')
df = df.append(test, sort=True)
df['ID'] = df.ID.fillna(-1).astype('int32')
df['year'] = df.date_block_num.map(year)
df['month'] = df.date_block_num.map(month)
df['category'] = df.item_id.map(category)
df['meta_category'] = df.category.map(meta_category)
df['city'] = df.shop_id.map(city)
train['category'] = train.item_id.map(category)

gb = train.groupby(by=groupby_cols, as_index=False).agg({'item_cnt_day': ['sum']})
gb.columns = [val[0] if val[-1] == '' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum': 'target'}, inplace=True)
df = pd.merge(df, gb, how='left', on=groupby_cols)

gb = train.groupby(by=['date_block_num', 'item_id'], as_index=False).agg({'item_cnt_day': ['sum']})
gb.columns = [val[0] if val[-1] == '' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum': 'target_item'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'item_id'])

gb = train.groupby(by=['date_block_num', 'shop_id'], as_index=False).agg({'item_cnt_day': ['sum']})
gb.columns = [val[0] if val[-1] == '' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum': 'target_shop'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'shop_id'])

gb = train.groupby(by=['date_block_num', 'category'], as_index=False).agg({'item_cnt_day': ['sum']})
gb.columns = [val[0] if val[-1] == '' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_cnt_day_sum': 'target_category'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'category'])

gb = train.groupby(by=['date_block_num', 'item_id'], as_index=False).agg({'item_price': ['mean', 'max']})
gb.columns = [val[0] if val[-1] == '' else '_'.join(val) for val in gb.columns.values]
gb.rename(columns={'item_price_mean': 'target_price_mean', 'item_price_max': 'target_price_max'}, inplace=True)
df = pd.merge(df, gb, how='left', on=['date_block_num', 'item_id'])
df['target_price_mean'] = np.minimum(df['target_price_mean'], df['target_price_mean'].quantile(0.99))
df['target_price_max'] = np.minimum(df['target_price_max'], df['target_price_max'].quantile(0.99))

df.fillna(0, inplace=True)
df['target'] = df['target'].clip(0, 20)
df['target_zero'] = (df['target'] > 0).astype('int32')
%%time

for enc_cols in [['shop_id', 'category'], ['shop_id', 'item_id'], ['shop_id'], ['item_id']]:

    col = '_'.join(['enc', *enc_cols])
    col2 = '_'.join(['enc_max', *enc_cols])
    df[col] = np.nan
    df[col2] = np.nan

    for d in tqdm_notebook(df.date_block_num.unique()):
        f1 = df.date_block_num < d
        f2 = df.date_block_num == d

        gb = df.loc[f1].groupby(enc_cols)[['target']].mean().reset_index()
        enc = df.loc[f2][enc_cols].merge(gb, on=enc_cols, how='left')[['target']].copy()
        enc.set_index(df.loc[f2].index, inplace=True)
        df.loc[f2, col] = enc['target']

        gb = df.loc[f1].groupby(enc_cols)[['target']].max().reset_index()
        enc = df.loc[f2][enc_cols].merge(gb, on=enc_cols, how='left')[['target']].copy()
        enc.set_index(df.loc[f2].index, inplace=True)
        df.loc[f2, col2] = enc['target']
def downcast(df):
    float32_cols = [c for c in df if df[c].dtype == 'float64']
    int32_cols = [c for c in df if df[c].dtype in ['int64', 'int16', 'int8']]

    df[float32_cols] = df[float32_cols].astype(np.float32)
    df[int32_cols] = df[int32_cols].astype(np.int32)

    return df
df.fillna(0, inplace=True)
df = downcast(df)
%%time

shift_range = [1, 2, 3, 4, 5, 12]

shifted_columns = [c for c in df if 'target' in c]

for shift in tqdm_notebook(shift_range):
    shifted_data = df[groupby_cols + shifted_columns].copy()
    shifted_data['date_block_num'] = shifted_data['date_block_num'] + shift

    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in shifted_columns else x
    shifted_data = shifted_data.rename(columns=foo)

    df = pd.merge(df, shifted_data, how='left', on=groupby_cols).fillna(0)
    df = downcast(df)

    del shifted_data
    gc.collect()
    sleep(1)
df = downcast(df)
drop_columns = [c for c in df if c[-1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'] and c.startswith('target')]
drop_columns += ['ID']
features = df.columns.difference(drop_columns)
f0 = df.date_block_num < 34
f1 = df.date_block_num == 34

train, val = train_test_split(df[f0], test_size=0.2, stratify=df[f0]['target'])
test = df[f1]

Train = xgb.DMatrix(train[features], train['target'])
Val = xgb.DMatrix(val[features], val['target'])
Test = xgb.DMatrix(test[features])
del df
gc.collect()
%%time

xgb_params = {
    'eval_metric': 'rmse',
    'lambda': '0.171', 
    'gamma': '0.124',
    'booster': 'gbtree', 
    'alpha': '0.170',
    'objective': 'reg:squarederror',
    'colsample_bytree': '0.715',
    'subsample': '0.874', 
    'silent': True,
    'min_child_weight': 26,
    'eta': '0.148',
    'max_depth': 6,
    'tree_method': 'gpu_hist', 
    'n_gpus': 1
}


model = xgb.train(xgb_params, Train, 1500, [(Train, 'Train'), (Val, 'Val')], early_stopping_rounds=10, verbose_eval=1)
test['item_cnt_month'] = model.predict(Test).clip(0, 20)

test[['ID', 'item_cnt_month']].sort_values('ID').to_csv('submission.csv', index=False)
pickle.dump(model, open('xgb.pickle', 'wb'))