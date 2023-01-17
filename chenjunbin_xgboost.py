# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from itertools import product

import gc

import seaborn as sns

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

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

    if 'date_block_num' in df.columns:

        df['date_block_num'] = df['date_block_num'].astype(np.int8)

    if 'shop_id' in df.columns:

        df['shop_id'] = df['shop_id'].astype(np.int8)

    if 'item_id' in df.columns:

        df['item_id'] = df['item_id'].astype(np.int16)

    if 'year' in df.columns:

        df['year'] = df['year'].astype(np.int8)

    if 'month' in df.columns:

        df['month'] = df['month'].astype(np.int8)

    if 'quarter' in df.columns:

        df['quarter'] = df['quarter'].astype(np.int8)

    return df
DATA_FOLDER     = '/kaggle/input/competitive-data-science-predict-future-sales/'



sales_train     = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))

items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))

item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))

shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv')).set_index('ID')
shops.head()
shops[shops.shop_id == 57]
items.head()
sales_train.describe()
sns.boxplot(x=sales_train['item_price'])
sns.boxplot(x=sales_train['item_cnt_day'])
sales_train.isnull().describe()
items.head()
item_categories.head()
shops.head()
test.head()
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
median = sales_train[(sales_train.shop_id==32)&(sales_train.item_id==2973)&(sales_train.date_block_num==4)&(sales_train.item_price>0)].item_price.median()

sales_train.loc[sales_train.item_price<0, 'item_price'] = median
sales_train = sales_train[sales_train['item_price']<30000]

sales_train = sales_train[sales_train['item_cnt_day']<1001]
median = sales_train[(sales_train['shop_id']==32) & (sales_train['item_id']==2973) & (sales_train['date_block_num']==4) 

                     & (sales_train['item_price']>0)].item_price.median()

sales_train.loc[sales_train['item_price']<0,'item_price'] = median
sales_train.date = pd.to_datetime(sales_train.date,format ="%d.%m.%Y")

sales_train['year'] = sales_train.date.dt.year

sales_train['month'] = sales_train.date.dt.month

sales_train['day'] = sales_train.date.dt.day
def date_agg_year(df):

    if df == 2013 :

        return  0

    if df == 2014 :

        return  1

    if df == 2015 :

        return  2

        

def date_agg_month(df):

    if df in [1,2,3]:

        return 1

    if df in [4,5,6]:

        return 2

    if df in [7,8,9]:

        return 3 

    if df in [10,11,12]:

        return 4
sales_train.year = sales_train.year.agg(date_agg_year)

sales_train['quarter'] = sales_train.month.agg(date_agg_month)
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



item_categories['split'] = item_categories['item_category_name'].str.split('-')

item_categories['type'] = item_categories['split'].map(lambda x: x[0].strip())

item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

# if subtype is nan then type

item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])

item_categories = item_categories[['item_category_id','type_code', 'subtype_code']]



items.drop(['item_name'], axis=1, inplace=True)
sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']
index_cols = ['shop_id','item_id','date_block_num']



grid = [] 



for block_num in sales_train['date_block_num'].unique():

    cur_shops = sales_train.loc[sales_train['date_block_num'] == block_num, 'shop_id'].unique()

    cur_items = sales_train.loc[sales_train['date_block_num'] == block_num, 'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops,cur_items,[block_num]])),dtype='int32'))



grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)



gb = sales_train.groupby(index_cols).agg({'item_cnt_day':['sum']})

gb.columns = ['target']

gb.reset_index(inplace=True)

all_data = pd.merge(grid, gb, on=index_cols, how='left').fillna(0)

all_data['target'] = all_data['target'].astype(np.float16)



gb = sales_train.groupby(['shop_id','date_block_num']).agg({'item_cnt_day':['sum']})

gb.columns = ['target_shop']

gb.reset_index(inplace=True)

all_data = pd.merge(all_data, gb, on=['shop_id','date_block_num'], how='left').fillna(0)

all_data['target_shop'] = all_data['target_shop'].astype(np.float16)



gb = sales_train.groupby(['item_id','date_block_num']).agg({'item_cnt_day':['sum']}) 

gb.columns = ['target_item']

gb.reset_index(inplace=True)

all_data = pd.merge(all_data, gb, on=['item_id','date_block_num'], how='left').fillna(0)

all_data['target_item'] = all_data['target_item'].astype(np.float16)



del gb

all_data = downcast_dtypes(all_data)



gc.collect()



all_data.sort_values(index_cols,inplace=True)
for i in range(12):

    all_data.loc[all_data.date_block_num==i,'year'] = 0

    all_data.loc[all_data.date_block_num==i,'month'] = i

for i in range(12,24):

    all_data.loc[all_data.date_block_num==i,'year'] = 1

    all_data.loc[all_data.date_block_num==i,'month'] = i-11

for i in range(24,35):

    all_data.loc[all_data.date_block_num==i,'year'] = 2

    all_data.loc[all_data.date_block_num==i,'month'] = i-23



all_data['quarter'] = all_data['month'].agg(date_agg_month)
#gb = all_data.groupby(['shop_id','item_id','quarter','year']).agg({'target':['mean']})

#gb.columns = ['quarter_mean_target']

#gb.reset_index(inplace=True)

#all_data = pd.merge(all_data,gb,on = ['shop_id','item_id','quarter','year'],how = 'left').fillna(0)

#all_data['quarter_mean_target'] = all_data['quarter_mean_target'].astype(np.float16)



#gb = all_data.groupby(['item_id','quarter','year']).agg({'target':['mean']})

#gb.columns = ['quarter_mean_target_item']

#gb.reset_index(inplace=True)

#all_data = all_data.merge(gb,on = ['item_id','quarter','year'],how = 'left').fillna(0)

#all_data['quarter_mean_target_item'] = all_data['quarter_mean_target_item'].astype(np.float16)



#gb = all_data.groupby(['shop_id','quarter','year']).agg({'target':['mean']})

#gb.columns = ['quarter_mean_target_shop']

#gb.reset_index(inplace=True)

#all_data = all_data.merge(gb,on = ['shop_id','quarter','year'],how = 'left').fillna(0)

#all_data['quarter_mean_target_shop'] = all_data['quarter_mean_target_shop'].astype(np.float16)



#all_data = downcast_dtypes(all_data)



#del gb

#gc.collect()



#all_data.sort_values(index_cols,inplace=True)
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)

all_data = pd.concat([all_data, test], ignore_index=True, sort=False, keys=index_cols).fillna(0)
for i in range(12):

    all_data.loc[all_data.date_block_num==i,'year'] = 0

    all_data.loc[all_data.date_block_num==i,'month'] = i

for i in range(12,24):

    all_data.loc[all_data.date_block_num==i,'year'] = 1

    all_data.loc[all_data.date_block_num==i,'month'] = i-11

for i in range(24,35):

    all_data.loc[all_data.date_block_num==i,'year'] = 2

    all_data.loc[all_data.date_block_num==i,'month'] = i-23



all_data['quarter'] = all_data['month'].agg(date_agg_month)
all_data = pd.merge(all_data, shops, on=['shop_id'], how='left')

sales_train = pd.merge(sales_train,shops,on='shop_id',how='left')

all_data = pd.merge(all_data, items, on=['item_id'], how='left')

all_data = pd.merge(all_data, item_categories, on=['item_category_id'], how='left')

all_data['city_code'] = all_data['city_code'].astype(np.int8)

all_data['item_category_id'] = all_data['item_category_id'].astype(np.int8)

all_data['type_code'] = all_data['type_code'].astype(np.int8)

all_data['subtype_code'] = all_data['subtype_code'].astype(np.int8)
gb = all_data.groupby(['item_category_id','date_block_num']).agg({'target':['sum']}) 

gb.columns = ['target_category']

gb.reset_index(inplace=True)

all_data = pd.merge(all_data, gb, on=['item_category_id','date_block_num'], how='left').fillna(0)

all_data['target_category'] = all_data['target_category'].astype(np.float16)
gb = all_data.groupby(['item_category_id']).agg({'item_id':['nunique']})

gb.columns =['item_category_num_count']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on = 'item_category_id',how='left').fillna(0)



gb = all_data.groupby(['shop_id']).agg({'item_id':['nunique']})

gb.columns =['shop_item_num_count']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on ='shop_id',how='left').fillna(0)



gb = all_data.groupby(['shop_id']).agg({'item_id':['nunique']})

gb.columns =['shop_item_category_num_count']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on ='shop_id',how='left').fillna(0)



gb = all_data.groupby(['city_code']).agg({'shop_id':['nunique']})

gb.columns =['city_shop_count']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on ='city_code',how='left').fillna(0)



gb = all_data.groupby(['type_code']).agg({'subtype_code':['nunique']})

gb.columns =['item_type_sub_count']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on ='type_code',how='left').fillna(0)



gb = all_data.groupby(['date_block_num','shop_id']).agg({'item_id':['nunique']})

gb.columns =['date_shop_item_count']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on =['date_block_num','shop_id'],how='left').fillna(0)



del gb

gc.collect()
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        gc.collect()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left').fillna(0)

    return df
all_data = lag_feature(all_data, [1,2,3,6,12], 'target')

all_data = lag_feature(all_data, [1,2,3,6,12], 'target_shop')

all_data = lag_feature(all_data, [1,2,3,6,12], 'target_item')

all_data = lag_feature(all_data, [1,2,3,6,12], 'target_category')
#all_data = lag_feature(all_data, [3,6,12], 'quarter_mean_target')

#all_data = lag_feature(all_data, [3,6,12], 'quarter_mean_target_item')

#all_data = lag_feature(all_data, [3,6,12], 'quarter_mean_target_shop')
gb = sales_train.groupby(['item_id']).agg({'item_price':['mean']})

gb.columns =['item_price_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['item_id'],how= 'left').fillna(0)

all_data['item_price_mean'] = all_data['item_price_mean'].astype(np.float16)



gb = sales_train.groupby(['item_id','date_block_num']).agg({'item_price':['mean']})

gb.columns =['date_item_price_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['item_id','date_block_num'],how= 'left').fillna(0)

all_data['date_item_price_mean'] = all_data['date_item_price_mean'].astype(np.float16)



gb = sales_train.groupby(['shop_id']).agg({'revenue':['mean']})

gb.columns =['shop_revenue_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['shop_id'],how= 'left').fillna(0)
gb = all_data.groupby(['item_id','date_block_num','city_code']).agg({'target':['mean']})

gb.columns =['date_city_item_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['item_id','date_block_num','city_code'],how= 'left').fillna(0)

all_data['date_city_item_target_mean'] = all_data['date_city_item_target_mean'].astype(np.float16)
gb = all_data.groupby(['shop_id','date_block_num','city_code']).agg({'target':['mean']})

gb.columns =['date_city_shop_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['shop_id','date_block_num','city_code'],how= 'left').fillna(0)

all_data['date_city_shop_target_mean'] = all_data['date_city_shop_target_mean'].astype(np.float16)
gb = all_data.groupby(['date_block_num','city_code']).agg({'target':['mean']})

gb.columns =['date_city_target_mean']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on=['date_block_num','city_code'],how= 'left').fillna(0)

all_data['date_city_target_mean'] = all_data['date_city_target_mean'].astype(np.float16)
gb = all_data.groupby(['type_code','date_block_num','city_code']).agg({'target':['mean']})

gb.columns =['date_city_type_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['type_code','date_block_num','city_code'],how= 'left').fillna(0)

all_data['date_city_type_target_mean'] = all_data['date_city_type_target_mean'].astype(np.float16)
gb = all_data.groupby(['item_category_id','date_block_num','city_code']).agg({'target':['mean']})

gb.columns =['date_city_category_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['item_category_id','date_block_num','city_code'],how= 'left').fillna(0)

all_data['date_city_category_target_mean'] = all_data['date_city_category_target_mean'].astype(np.float16)
gb = all_data.groupby(['item_id','date_block_num']).agg({'target':['mean']})

gb.columns =['date_item_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['item_id','date_block_num'],how= 'left').fillna(0)

all_data['date_item_target_mean'] = all_data['date_item_target_mean'].astype(np.float16)
gb = all_data.groupby(['type_code','date_block_num']).agg({'target':['mean']})

gb.columns =['date_type_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['type_code','date_block_num'],how= 'left').fillna(0)

all_data['date_type_target_mean'] = all_data['date_type_target_mean'].astype(np.float16)
gb = all_data.groupby(['shop_id','date_block_num']).agg({'target':['mean']})

gb.columns =['date_shop_target_mean']

gb.reset_index(inplace = True)

all_data = all_data.merge(gb,on=['shop_id','date_block_num'],how= 'left').fillna(0)

all_data['date_shop_target_mean'] = all_data['date_shop_target_mean'].astype(np.float16)
gb = all_data.groupby(['item_category_id','date_block_num']).agg({'target':['mean']})

gb.columns =['date_category_target_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['item_category_id','date_block_num'],how= 'left').fillna(0)

all_data['date_category_target_mean'] = all_data['date_category_target_mean'].astype(np.float16)
gb = sales_train.groupby(['shop_id','date_block_num']).agg({'revenue':['sum']})

gb.columns =['date_shop_revenue']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['shop_id','date_block_num'],how= 'left').fillna(0)
gb = sales_train.groupby(['shop_id','date_block_num']).agg({'revenue':['mean']})

gb.columns =['date_shop_revenue_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['shop_id','date_block_num'],how= 'left').fillna(0)
gb = sales_train.groupby(['date_block_num','city_code']).agg({'revenue':['mean']})

gb.columns =['date_city_revenue_mean']

gb.reset_index(inplace = True)

all_data = pd.merge(all_data,gb,on=['date_block_num','city_code'],how= 'left').fillna(0)



del gb

gc.collect()
all_data = lag_feature(all_data, [1], 'date_item_price_mean')
all_data = lag_feature(all_data, [1], 'date_shop_item_count')
all_data = lag_feature(all_data, [1,2,3], 'date_shop_revenue')
all_data = lag_feature(all_data,[1],'date_shop_revenue_mean')
all_data = lag_feature(all_data,[1], 'date_city_type_target_mean')
all_data = lag_feature(all_data,[1], 'date_city_item_target_mean')
all_data = lag_feature(all_data,[1], 'date_city_shop_target_mean')
all_data = lag_feature(all_data,[1], 'date_city_category_target_mean')
all_data = lag_feature(all_data,[1], 'date_city_target_mean')
all_data = lag_feature(all_data,[1], 'date_item_target_mean')
all_data = lag_feature(all_data,[1], 'date_shop_target_mean')
all_data = lag_feature(all_data,[1], 'date_category_target_mean')
all_data = lag_feature(all_data,[1], 'date_city_revenue_mean')
all_data = lag_feature(all_data,[1], 'date_shop_revenue_mean')
all_data = all_data[all_data.date_block_num > 12]

all_data = downcast_dtypes(all_data)
all_data.to_pickle('data.pkl')
del all_data

del shops

del items

del item_categories



gc.collect()
all_data = pd.read_pickle('data.pkl')
all_data.columns
all_data = downcast_dtypes(all_data)
columns = ['shop_id', 'item_id', 'date_block_num', 

           'year', 'month', 'quarter', 'city_code',

       'item_category_id', 'type_code', 'subtype_code',

#       'item_category_num_count', 'shop_item_num_count',

#       'shop_item_category_num_count', 'city_shop_count',

#       'item_type_sub_count', 'date_shop_item_count',

       'target_lag_1',

       'target_lag_2', 'target_lag_3', 'target_lag_6', 'target_lag_12',

       'target_shop_lag_1', 'target_shop_lag_2', 'target_shop_lag_3',

       'target_shop_lag_6', 'target_shop_lag_12', 'target_item_lag_1',

       'target_item_lag_2', 'target_item_lag_3', 'target_item_lag_6',

       'target_item_lag_12','target_category_lag_1','target_category_lag_2','target_category_lag_3',

       'target_category_lag_6','target_category_lag_12','item_price_mean', 

       'date_item_price_mean_lag_1', 'date_shop_item_count_lag_1',

       'date_shop_revenue_lag_1', 'date_shop_revenue_lag_2','date_shop_revenue_lag_3',

       'date_shop_revenue_mean_lag_1',

       'date_city_type_target_mean_lag_1',

       'date_city_item_target_mean_lag_1', 'date_city_shop_target_mean_lag_1',

       'date_city_category_target_mean_lag_1', 'date_city_target_mean_lag_1',

       'date_item_target_mean_lag_1', 'date_shop_target_mean_lag_1',

       'date_category_target_mean_lag_1', 'date_city_revenue_mean_lag_1'

          ]
train_x = all_data.loc[all_data.date_block_num < 33, columns]

train_y = all_data.loc[all_data.date_block_num < 33,'target']
train_x.info()
validation_x = all_data.loc[all_data.date_block_num == 33, columns]

validation_y = all_data.loc[all_data.date_block_num == 33,'target']
test_x = all_data.loc[all_data.date_block_num == 34, columns]
del all_data



gc.collect()
model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8,

    eta=0.3,    

    seed=42)



model.fit(

    train_x, 

    train_y,

    eval_metric="rmse",

    eval_set=[(train_x, train_y), (validation_x, validation_y)],

    verbose=True,

    early_stopping_rounds = 10)
plot_features(model, (10,14))
Y_validation = model.predict(validation_x).clip(0, 20)

Y_test = model.predict(test_x).clip(0, 20)
submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)
submission.head()