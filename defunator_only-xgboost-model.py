from google.colab import files
files.upload() #upload kaggle.json

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json

!kaggle competitions download -c competitive-data-science-predict-future-sales

!unzip -q sales_train.csv.zip -d .
!unzip -q sample_submission.csv.zip -d .
!unzip -q items.csv.zip -d .
!unzip -q test.csv.zip -d .
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
%matplotlib inline

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
plt.figure(figsize=(15,4))
sns.boxplot(train['item_price'])

plt.figure(figsize=(15,4))
sns.boxplot(train['item_cnt_day'])
train = train[train['item_price'].lt(100000) & train['item_cnt_day'].lt(1000)]
shops[['shop_name', 'shop_id']].drop_duplicates().sort_values(by=['shop_name'])
train.loc[train['shop_id'].eq(0),'shop_id'] = 57
test.loc[test['shop_id'].eq(0),'shop_id'] = 57
# shops.loc[shops['shop_id'].eq(0),'shop_id'] = 57

train.loc[train['shop_id'].eq(1),'shop_id'] = 58
test.loc[test['shop_id'].eq(1),'shop_id'] = 58
# shops.loc[shops['shop_id'].eq(1),'shop_id'] = 58

train.loc[train['shop_id'].eq(10),'shop_id'] = 11
test.loc[test['shop_id'].eq(10),'shop_id'] = 11
# shops.loc[shops['shop_id'].eq(10),'shop_id'] = 11
test.loc[:,'date_block_num'] = 34
test.loc[:,'month'] = 11
train.loc[:,'month'] = train.loc[:,'date'].apply(lambda x: int(x.split('.')[1]))
keys = ['date_block_num', 'month', 'shop_id', 'item_id']
train = train.groupby(keys).agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index().rename(columns={'item_cnt_day': 'item_cnt_month'})
train.loc[:,'item_cnt_month'] = train['item_cnt_month'].clip(0, 20)
shops.loc[:,'is_online'] = 0
shops.loc[shops['shop_id'].eq(55) | shops['shop_id'].eq(12), 'is_online'] = 1
n_online = np.unique(train['shop_id'], return_counts=True)[1][55 - 1] + np.unique(train['shop_id'], return_counts=True)[1][12 - 1]
n_total = train.shape[0]
print('online buys fraction =', n_online / n_total)
from sklearn.preprocessing import LabelEncoder

shops.loc[:,'shop_city'] = shops.loc[:,'shop_name'].apply(lambda x: x.split()[0])
shops.loc[shops['shop_city'].eq('!Якутск'),'shop_city'] = 'Якутск'
shops.loc[:,'shop_city'] = LabelEncoder().fit_transform(shops['shop_city'])
items.loc[:,'name1'], items.loc[:,'name2'] = items['item_name'].str.split('[', 1).str
items.loc[:,'name1'], items.loc[:,'name3'] = items['item_name'].str.split('(', 1).str

items.loc[:,'name2'] = items['name2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items.loc[:,'name3'] = items['name3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()

items.fillna('0', inplace=True)

items.loc[:,'name2'] = items['name2'].apply(lambda x: x[:-1] if x != '0' else x)

items.loc[:,'type'] = items['name2'].apply(lambda x: x[0:8] if x.split(' ')[0] == 'xbox' else x.split(' ')[0])
items.loc[items['type'].eq('x360') | items['type'].eq('xbox360') | items['type'].eq('xbox 360'),'type'] = 'xbox 360'
items.loc[items['type'].eq(''), 'type'] = 'mac'
items.loc[items['type'].eq('pc') | items['type'].eq('рс') | items['type'].eq('pс'),'type'] = 'pc'
items.loc[items['type'].eq('цифровая') | items['type'].eq('цифро'), 'type'] = 'цифро'
items.loc[items['type'].eq('kg4') | items['type'].eq('5c5') |
          items['type'].eq('5c7') | items['type'].eq('kf7') | items['type'].eq('kf6'),'type'] = 'xboxone'
items.loc[:,'type'] = items.type.apply( lambda x: x.replace(' ', '') )
items['type'].unique()
strange_codes = ['6jv', 'j72', 'hm3', 's3v', '6dv', '6l6', '5f4', 's4v', 'kg4', '5c5', '5c7', 'kf7', 'kf6']
got = set()
for x in items['item_name'].unique():
    if len(x.split('[')) < 2:
        continue
    for code in strange_codes: 
        if code in x.split('[')[1].lower():
            print(x)
            got.add(code)
print(set(strange_codes) - got)
xboxone_consoles = ['kg4', '5c5', '5c7', 'kf7', 'kf6']
group_sum = items.groupby(['type']).agg({'item_id': 'count'}).reset_index()
drop_cols = []

for cat in group_sum['type'].unique():
    if group_sum.loc[group_sum['type'].eq(cat), "item_id"].values[0] < 40:
        drop_cols.append(cat)

items.loc[:,'name2'] = items.loc[:,'name2'].apply(lambda x: 'other' if (x in drop_cols) else x)
items.loc[:,'name2'] = LabelEncoder().fit_transform(items['name2'])
items.loc[:,'name3'] = LabelEncoder().fit_transform(items['name3'])
items.loc[:,'type'] = LabelEncoder().fit_transform(items['type'])

item_categories.loc[:,'cat_global'] = np.select([
    item_categories.item_category_id.isin(range(0,8)),
    item_categories.item_category_id.isin([8,80]),
    item_categories.item_category_id.eq(9),
    item_categories.item_category_id.isin(range(10,18)),
    item_categories.item_category_id.isin(range(18,32)),
    item_categories.item_category_id.isin([32,33,34,35,36,37,79]),
    item_categories.item_category_id.isin(range(37,42)),
    item_categories.item_category_id.isin(range(42,55)),
    item_categories.item_category_id.isin(range(55,61)),
    item_categories.item_category_id.isin(range(61,73)),
    item_categories.item_category_id.isin(range(73,79)),
    item_categories.item_category_id.isin([81,82]),
    item_categories.item_category_id.eq(83),
    item_categories.item_category_id.eq(84)
    ], [
    'accessories','tickets','delivery','consoles','games',
    'payment_cards','movies','books','music','gifts','programs',
    'discs','batteries','plastic_bags'
])

item_categories.loc[:,'cat_platform'] = np.select([
    item_categories.item_category_name.str.contains('PS2', case=False),
    item_categories.item_category_name.str.contains('PS3', case=False),
    item_categories.item_category_name.str.contains('PS4', case=False),
    item_categories.item_category_name.str.contains('PSP', case=False),
    item_categories.item_category_name.str.contains('PSVita', case=False),
    item_categories.item_category_name.str.contains('XBOX 360', case=False),
    item_categories.item_category_name.str.contains('XBOX ONE', case=False),
    item_categories.item_category_name.str.contains('PC', case=False),
    item_categories.item_category_name.str.contains('MAC', case=False),
    item_categories.item_category_name.str.contains('Android', case=False)],
    ['PS2','PS3','PS4','PSP','PSVita','XBOX_360','XBOX_ONE','PC','MAC','Android'],
    default='other')
item_categories.loc[:,'subtype'] = item_categories['item_category_name'].apply(lambda x: x.split('-')[-1].strip())
item_categories.loc[:,'subtype'] = LabelEncoder().fit_transform(item_categories['subtype'])
item_categories.loc[:,'cat_global'] = LabelEncoder().fit_transform(item_categories['cat_global'])
item_categories.loc[:,'cat_platform'] = LabelEncoder().fit_transform(item_categories['cat_platform'])
print('unique test item_id =', test['item_id'].nunique())
print('unique test shop_id =', test['shop_id'].nunique())
print('#item_ids * #shop_ids =', test['item_id'].nunique() * test['shop_id'].nunique())
print('test size =', test.shape[0])
trues = 0
for date_block_num in train['date_block_num'].unique():
    item_ids, shop_ids = train['item_id'].nunique(), train['shop_id'].nunique()
    if item_ids * shop_ids == train[train['date_block_num'].eq(date_block_num)].shape[0]:
        trues += 1
        
print('It is true for', trues, 'date_block_nums')
from itertools import product

df = []
for date_num in train['date_block_num'].unique():
  date_num_sales = train[train['date_block_num'].eq(date_num)]
  month = date_num_sales['month'].iloc[0]
  df.append(list(product([date_num], [month], date_num_sales['item_id'].unique(), date_num_sales['shop_id'].unique())))

df = pd.DataFrame(np.vstack(df), columns=['date_block_num', 'month', 'item_id', 'shop_id'])
test.loc[:,'item_cnt_month'] = 0
test.loc[:,'item_price'] = 0
train.loc[:,'ID'] = -999

# add info from train
cols = ['date_block_num', 'month', 'item_id', 'shop_id', 'item_cnt_month', 'ID', 'item_price']
df = df.merge(train[cols], on=['date_block_num', 'month', 'item_id', 'shop_id'], how='left')

df = pd.concat([df, test[cols]])
df.sort_values(['date_block_num', 'month', 'shop_id', 'item_id'], inplace=True)
df = df.merge(shops[['shop_id','shop_city','is_online']], on=['shop_id'], how='left')
df = df.merge(items[['item_id','item_category_id','name2','name3','type']], on=['item_id'], how='left')
df = df.merge(item_categories[['item_category_id','cat_global','cat_platform','subtype']], on=['item_category_id'], how='left')

df['ID'].fillna(-999, inplace=True)
df['item_cnt_month'].fillna(0, inplace=True)
df['item_price'].fillna(0, inplace=True)
df.isna().any()
train = train.merge(items[['item_id','item_category_id','name2','name3','type']], on=['item_id'], how='left')
train = train.merge(item_categories[['item_category_id','cat_global','cat_platform','subtype']], on=['item_category_id'], how='left')

for date_num in df.date_block_num.unique():
    for feature in ['item_category_id', 'name2', 'name3', 'type', 'subtype', 'cat_global', 'cat_platform']:
        mean_price = train[train.date_block_num.ne(date_num)].groupby(feature).item_price.mean()
        df.loc[df.date_block_num.eq(date_num),f'{feature}_mean_price'] = df.loc[df.date_block_num.eq(date_num),feature].map(mean_price)
        
for feature in ['item_category_id', 'name2', 'name3', 'type', 'subtype', 'cat_global', 'cat_platform']:
    mean_price = train.groupby(feature).item_price.mean()
    df.loc[df[f'{feature}_mean_price'].isna(),f'{feature}_mean_price'] = df.loc[df[f'{feature}_mean_price'].isna(),feature].map(mean_price)
df[df.date_block_num.ne(34)].isna().sum()
df[df.date_block_num.eq(34)].isna().sum()
df.nunique()
df['shop_city'] = df['shop_city'].astype(np.int8)
df['item_category_id'] = df['item_category_id'].astype(np.int8)
df['date_block_num'] = df['date_block_num'].astype(np.int8)
df['item_id'] = df['item_id'].astype(np.int16)
df['shop_id'] = df['shop_id'].astype(np.int8)
df['month'] = df['month'].astype(np.int8)
df['item_cnt_month'] = df['item_cnt_month'].astype(np.float16)
df['item_price'] = df['item_price'].astype(np.float32)
df['subtype'] = df['subtype'].astype(np.int8)
df['ID'] = df['ID'].astype(np.int32)
df['is_online'] = df['is_online'].astype(np.int8)
df['name2'] = df['name2'].astype(np.int16)
df['name3'] = df['name3'].astype(np.int16)
df['type'] = df['type'].astype(np.int8)
df['cat_global'] = df['cat_global'].astype(np.int8)
df['cat_platform'] = df['cat_platform'].astype(np.int8)
df['item_category_id_mean_price'] = df['item_category_id_mean_price'].astype(np.float32)
df['name2_mean_price'] = df['name2_mean_price'].astype(np.float32)
df['type_mean_price'] = df['type_mean_price'].astype(np.float32)
df['subtype_mean_price'] = df['subtype_mean_price'].astype(np.float32)
df['cat_global_mean_price'] = df['cat_global_mean_price'].astype(np.float32)
df['cat_platform_mean_price'] = df['cat_platform_mean_price'].astype(np.float32)
df['name3_mean_price'] = df['name3_mean_price'].astype(np.float32)

# del test, train
def create_lagged(df, feature, lags):
  keys = ['date_block_num', 'shop_id', 'item_id']
  df_cp = df[keys + [feature]].copy()
  for lag in lags:
    df_cp = df_cp.rename(columns={feature: f'{feature}_{lag}'})
    feature = f'{feature}_{lag}'
    df_cp.loc[:,'date_block_num'] += lag
    df = df.merge(df_cp, on=keys, how='left')
    df_cp.loc[:,'date_block_num'] -= lag

  del df_cp

  return df
import gc

def select_id(row):
  for i in range(3):
    if row.iloc[i]:
      return row.iloc[i]
  return 0

df = create_lagged(df, 'item_cnt_month', [1,2,3])
# df.loc[:,'item_cnt_month_lag'] = df[['item_cnt_month_1', 'item_cnt_month_2', 'item_cnt_month_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num']
mean_month_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_month_cnt'})
df = df.merge(mean_month_group, on=ix, how='left')
df.loc[:,'mean_month_cnt'] = df['mean_month_cnt'].astype(np.float16)
del mean_month_group
df = create_lagged(df, 'mean_month_cnt', [1])
df.drop(['mean_month_cnt'], axis=1, inplace=True)
# df.loc[:,'mean_month_cnt_lag'] = df[['mean_month_cnt_1', 'mean_month_cnt_2', 'mean_month_cnt_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num', 'item_id']
mean_item_id_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_item_id_cnt'})
df = df.merge(mean_item_id_group, on=ix, how='left')
del mean_item_id_group
df = create_lagged(df, 'mean_item_id_cnt', [1,2,3])
# df.loc[:,'mean_item_id_cnt_lag'] = df[['mean_item_id_cnt_1', 'mean_item_id_cnt_2', 'mean_item_id_cnt_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num', 'item_category_id']
mean_cat_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_cat_cnt'})
df = df.merge(mean_cat_group, on=ix, how='left')
del mean_cat_group
df = create_lagged(df, 'mean_cat_cnt', [1,2])
# df.loc[:,'mean_cat_cnt_lag'] = df[['mean_cat_cnt_1', 'mean_cat_cnt_2', 'mean_cat_cnt_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num', 'item_category_id', 'shop_id']
mean_cat_shop_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_cat_shop_cnt'})
df = df.merge(mean_cat_shop_group, on=ix, how='left')
del mean_cat_shop_group
df = create_lagged(df, 'mean_cat_shop_cnt', [1,2,3])
# df.loc[:,'mean_cat_shop_cnt_lag'] = df[['mean_cat_shop_cnt_1', 'mean_cat_shop_cnt_2', 'mean_cat_shop_cnt_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num', 'shop_city', 'item_id']
mean_city_id_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_city_id_cnt'})
df = df.merge(mean_city_id_group, on=ix, how='left')
del mean_city_id_group
df = create_lagged(df, 'mean_city_id_cnt', [1,2])
# df.loc[:,'mean_city_id_cnt_lag'] = df[['mean_city_id_cnt_1', 'mean_city_id_cnt_2', 'mean_city_id_cnt_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num', 'shop_city', 'item_category_id']
mean_city_cat_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_city_cat_cnt'})
df = df.merge(mean_city_cat_group, on=ix, how='left')
del mean_city_cat_group
df = create_lagged(df, 'mean_city_cat_cnt', [1])
# df.loc[:,'mean_city_cat_cnt_lag'] = df[['mean_city_cat_cnt_1', 'mean_city_cat_cnt_2', 'mean_city_cat_cnt_3']].apply(select_id, axis=1)
gc.collect()
ix = ['date_block_num', 'item_id']
mean_id_price_group = df.groupby(ix).agg({'item_price': 'mean'}).reset_index().rename(columns={'item_price': 'mean_id_price'})
df = df.merge(mean_id_price_group, on=ix, how='left')
del mean_id_price_group
df = create_lagged(df, 'mean_id_price', [1,2,3])
# df.loc[:,'mean_id_price_lag'] = df[['mean_id_price_1', 'mean_id_price_2', 'mean_id_price_3']].apply(select_id, axis=1)
gc.collect()
df.loc[:,'mean_id_price_lag'] = df[['mean_id_price_1', 'mean_id_price_1_2', 'mean_id_price_1_2_3']].apply(select_id, axis=1)
mean_price_group = df.groupby(['item_id']).agg({'item_price': 'mean'}).reset_index().rename(columns={'item_price': 'mean_price'})
df = df.merge(mean_price_group, on=['item_id'], how='left')
del mean_price_group
df.loc[:,'delta_price_lag'] = (df['mean_id_price_lag'] - df['mean_price']) / df['mean_price']
gc.collect()
ix = ['date_block_num', 'shop_id', 'subtype']
mean_id_price_group = df.groupby(ix).agg({'item_cnt_month': 'mean'}).reset_index().rename(columns={'item_cnt_month': 'mean_shop_subtype_cnt'})
df = df.merge(mean_id_price_group, on=ix, how='left')
del mean_id_price_group
df = create_lagged(df, 'mean_shop_subtype_cnt', [1])
# df.loc[:,'mean_id_price_lag'] = df[['mean_id_price_1', 'mean_id_price_2', 'mean_id_price_3']].apply(select_id, axis=1)
gc.collect()
item_sale = df.groupby(['item_id'])['date_block_num']
df.loc[:,'item_first_sale'] = df['date_block_num'] - item_sale.transform('min')

item_shop_sale = df.groupby(['item_id', 'shop_id'])['date_block_num']
df.loc[:,'item_shop_first_sale'] = df['date_block_num'] - item_shop_sale.transform('min')
item_sales_sum = df.groupby(['item_id'])['item_cnt_month'].cumsum()
item_sales_cnt = df.groupby(['item_id'])['item_cnt_month'].cumcount()

train_mask, test_mask = df['date_block_num'] != 34, df['date_block_num'] == 34
df.loc[:,'item_id_mean_enc'] = item_sales_sum / item_sales_cnt
df.loc[test_mask,'item_id_mean_enc'] = np.nan

item_id_mean_enc_mean = df.loc[train_mask,:].groupby(['item_id'])['item_cnt_month'].mean()
df.loc[df['item_id_mean_enc'].isna(),'item_id_mean_enc'] = df['item_id'].map(item_id_mean_enc_mean)
df['item_id_mean_enc'].fillna(df.loc[train_mask,'item_id_mean_enc'].mean(), inplace=True)
# better check that there is no overfitting with item_price (because it is not always available for the test)
df.loc[:,'money_flow'] = df['item_price'] * df['item_cnt_month']

shop_id_money_sum = df.groupby(['date_block_num', 'shop_id'])['money_flow'].transform('sum')
shop_id_money_cnt = df.groupby(['date_block_num', 'shop_id'])['money_flow'].transform('count')

df.loc[train_mask,'cur_money_flow'] = df.groupby(['date_block_num', 'shop_id'])['money_flow'].transform('sum')
df.loc[:,'shop_id_mean_enc'] = (shop_id_money_sum - df['cur_money_flow']) / (shop_id_money_cnt - 1)

shop_id_mean_enc_mean = df.loc[train_mask,:].groupby(['shop_id'])['cur_money_flow'].mean()
df.loc[test_mask,'shop_id_mean_enc'] = df['shop_id'].map(shop_id_mean_enc_mean)
df.isna().any()
df.columns
# df = pd.read_hdf('./data.h5', 'df')
df.fillna(0, inplace=True)

target = 'item_cnt_month'
features = [
    'date_block_num',
    'month',
    'item_id',
    'shop_id',
    'shop_id_mean_enc',
    'shop_city',
    'is_online',
    'name2',
    'name3',
    'type',
    'item_category_id',
    'subtype',
    'item_cnt_month_1',
    'item_cnt_month_1_2',
    'item_cnt_month_1_2_3',
    'mean_month_cnt_1',
    'mean_item_id_cnt_1',
    'mean_item_id_cnt_1_2', 
    'mean_item_id_cnt_1_2_3',
    'mean_cat_cnt_1',
    'mean_cat_cnt_1_2',
    'mean_cat_shop_cnt_1',
    'mean_cat_shop_cnt_1_2',
    'mean_cat_shop_cnt_1_2_3',
    'mean_city_id_cnt_1',
    'mean_city_id_cnt_1_2',
    'mean_city_cat_cnt_1',
#     'mean_id_price_1',
#     'mean_id_price_1_2',
#     'mean_id_price_1_2_3',
#     'mean_id_price_lag',
#     'mean_price',
    'delta_price_lag',
#     'item_price_mean',
    'mean_shop_subtype_cnt_1',
    'item_first_sale',
    'item_shop_first_sale',
#     'cat_global',
#     'cat_platform',
    'item_category_id_mean_price',
#     'name2_mean_price',
#     'type_mean_price',
    'subtype_mean_price',
    'cat_global_mean_price',
    'cat_platform_mean_price',
#     'name3_mean_price'
]

train = df.loc[df['date_block_num'].ge(4) & df['date_block_num'].lt(33), features + [target]]
val = df.loc[df['date_block_num'].eq(33), features + [target]]
test = df.loc[df['date_block_num'].eq(34), features + ['ID']]

df.to_hdf('data.h5', 'df')
del df
gc.collect()
gc.collect()
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse

model = xgb.XGBRegressor(
    max_depth=8,
    n_estimators=1500,
    min_child_weight=400, 
    colsample_bytree=0.8, 
    subsample=0.8,
    eta=0.05,
    tree_method='gpu_hist',
    seed=42)
gc.collect()
model.fit(train[features], train[target], eval_metric="rmse", 
    eval_set=[(val[features], val[target])],
    early_stopping_rounds=100)
test.loc[:,'item_cnt_month'] = np.clip(model.predict(test[features]), 0, 20)
subm = test[['ID', 'item_cnt_month']]

subm.to_csv('submission.csv', index=False)
# !kaggle competitions submit -c competitive-data-science-predict-future-sales -f ./submission.csv -m "Everything fixed! I guess..."
# eta=0.05: val = 0.89882; train = 0.8048361; test = 0.897449 and 0.902729
# min_child_weight=500,early_stopping_rounds=100: val=0.88778; train=0.7782264; test=0.89203
# min_child_weight=1000: val=0.89282; train=0.80574894; test=0.89247
# min_child_weight=700: val=0.89248; train=0.798477; test=0.89223
# min_child_weight=400: val=0.88888; train=0.7788778; test=0.89099

# new dataset eta=0.01: val=0.89530104; train=0.7996507; test=0.89768
# new dataset eta=0.05: val=0.89576; train=0.79732686; test=0.88808
preds = np.clip(model.predict(train[features]), 0, 20)
print(np.sqrt(mse(train[target], preds)))
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))
len([
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    #'date_shop_type_avg_item_cnt_lag_1',
    #'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    #'date_type_avg_item_cnt_lag_1',
    #'date_subtype_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
])
from sklearn.preprocessing import LabelEncoder

df[features]
