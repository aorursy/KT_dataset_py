# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

# set index to ID to avoid droping it later

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
train.shape
train.head()
train.describe()
items.head()
shops.head()
cats.head()
fig, axes = plt.subplots(2, 1, figsize=(4, 4))

fig.set_dpi(100)



sns.boxplot(x=train.item_cnt_day, ax=axes[0])

sns.boxplot(x=train.item_price, ax=axes[1])



plt.tight_layout()
train = train[train.item_price<100000]

train = train[train.item_cnt_day<1001]
# item count day 商品单天销售数量存在负数的情况，退货？？？

train[(train.item_cnt_day<0)].head()
train[(train.item_price<0)]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
shops.head()
# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
# Сергиев Посад ТЦ "7Я" СергиевПосад ТЦ "7Я" 是一家店 去重

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'



# 第一个单词是城市名称

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

# 城市名称误写更正

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

# shops属性用数值化后的特征替代

shops = shops[['shop_id','city_code']]
shops.head()
items.head()
# 只保留商品数值化特征

items.drop(['item_name'], axis=1, inplace=True)
items.head()
cats.head()
cats['split'] = cats['item_category_name'].str.split('-')

cats['type'] = cats['split'].map(lambda x: x[0].strip())

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])



# 子类为空，则用大类填充

cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]
cats.head()
train.head()
# 将单天销售额作为训练特征

train['revenue'] = train['item_price']*train['item_cnt_day']
train.head()
# 注意同一家商店的同一商品价格可能随时间而变化的

train[(train.item_id==27) & (train.shop_id==2)]
from itertools import product

train_monthly = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train[train.date_block_num==i]

    train_monthly.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
train_monthly = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum'], 'revenue': ['sum']})
train_monthly.columns
train_monthly.columns=['item_cnt_month', 'revenue_monthly']
train_monthly.columns
train_monthly.reset_index(inplace=True)
# 比赛要求

train_monthly['item_cnt_month'].clip(0, 20)
train_monthly.head()
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)



train_monthly = pd.concat([train_monthly, test], ignore_index=True, sort=False, keys=['date_block_num','shop_id','item_id'])

train_monthly.fillna(0, inplace=True)
train_monthly = pd.merge(train_monthly, shops, on=['shop_id'], how='left')

train_monthly = pd.merge(train_monthly, items, on=['item_id'], how='left')

train_monthly = pd.merge(train_monthly, cats, on=['item_category_id'], how='left')
train_monthly.head()
train_monthly.info()
train_monthly['item_id'].nunique()
# 按照数据类型压缩数据集

train_monthly['date_block_num'] = train_monthly['date_block_num'].astype(np.int8)

train_monthly['shop_id'] = train_monthly['shop_id'].astype(np.int8)

train_monthly['item_id'] = train_monthly['item_id'].astype(np.int16)

train_monthly['city_code'] = train_monthly['city_code'].astype(np.int8)

train_monthly['item_category_id'] = train_monthly['item_category_id'].astype(np.int8)

train_monthly['type_code'] = train_monthly['type_code'].astype(np.int8)

train_monthly['subtype_code'] = train_monthly['subtype_code'].astype(np.int8)
train_monthly.info()
train_monthly['date_block_num'].unique()
def traceback_feature(df, feature, previous=[1, 12]):

    """

    引入 feature 特征在上个月，上个季度对应月，去年的当月的情况作为新特征

    """

    for i in previous:

        # 第 i 个月前的销售信息

        df_previous = df[['date_block_num','shop_id','item_id',feature]].copy()

        # 作为当月的第i月前销售特征

        df_previous.columns = ['date_block_num','shop_id','item_id','prev_{}_{}'.format(i, feature)]

        df_previous['date_block_num'] += i

        df = pd.merge(df, df_previous, on=['date_block_num','shop_id','item_id'], how='left')

    return df
train_monthly =  traceback_feature(train_monthly, 'item_cnt_month')
train_monthly[train_monthly['date_block_num']==0].head()
# for shop in train_monthly['shop_id'].unique():

#     for item in train_monthly['item_id'].unique():

#         for date_num in train_monthly['date_block_num'].unique():

            

#             # 计算date_num月份的过去3个月的月销售额总量

#             past_three_month = 0

#             for i in range(1, 4):

#                 # 避免date_block_num出现负数，没有意义

#                 past_date_num = max(date_num-i, 0)

                

#                 item_cnt_past = train_monthly[(train_monthly['shop_id']==shop) & 

#                           (train_monthly['item_id']==item) &

#                           (train_monthly['date_block_num']==past_date_num)]['item_cnt_month']

#                 past_three_month += item_cnt_past

            

#             # 计算得到的过去3个月的月销售额总量赋值给新的特征

#             train_monthly[(train_monthly['shop_id']==shop) & 

#                           (train_monthly['item_id']==item) &

#                           (train_monthly['date_block_num']==date_num-i)]['p_3_item_cnt_month']=past_three_month

            
train_monthly[(train_monthly['shop_id']==2) & 

                          (train_monthly['item_id']==317) &

                          (train_monthly['date_block_num']==0)]
avg_total_cnt_monthly = train_monthly.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})

avg_total_cnt_monthly.columns = ['avg_total_cnt_monthly']

avg_total_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_total_cnt_monthly, on=['date_block_num'], how='left')

train_monthly['avg_total_cnt_monthly'] = train_monthly['avg_total_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_total_cnt_monthly')
avg_shop_cnt_monthly = train_monthly.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})

avg_shop_cnt_monthly.columns = ['avg_shop_cnt_monthly']

avg_shop_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_shop_cnt_monthly, on=['date_block_num', 'shop_id'], how='left')

train_monthly['avg_shop_cnt_monthly'] = train_monthly['avg_shop_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_shop_cnt_monthly')
avg_item_cnt_monthly = train_monthly.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})

avg_item_cnt_monthly.columns = ['avg_item_cnt_monthly']

avg_item_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_item_cnt_monthly, on=['date_block_num', 'item_id'], how='left')

train_monthly['avg_item_cnt_monthly'] = train_monthly['avg_item_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_item_cnt_monthly')
avg_category_cnt_monthly = train_monthly.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})

avg_category_cnt_monthly.columns = ['avg_category_cnt_monthly']

avg_category_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_category_cnt_monthly, on=['date_block_num', 'item_category_id'], how='left')

train_monthly['avg_category_cnt_monthly'] = train_monthly['avg_category_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_category_cnt_monthly')
avg_type_cnt_monthly = train_monthly.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})

avg_type_cnt_monthly.columns = ['avg_type_cnt_monthly']

avg_type_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_type_cnt_monthly, on=['date_block_num', 'type_code'], how='left')

train_monthly['avg_type_cnt_monthly'] = train_monthly['avg_type_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_type_cnt_monthly')
avg_subtype_cnt_monthly = train_monthly.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})

avg_subtype_cnt_monthly.columns = ['avg_subtype_cnt_monthly']

avg_subtype_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_subtype_cnt_monthly, on=['date_block_num', 'subtype_code'], how='left')

train_monthly['avg_subtype_cnt_monthly'] = train_monthly['avg_subtype_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_subtype_cnt_monthly')
avg_shop_category_cnt_monthly = train_monthly.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})

avg_shop_category_cnt_monthly.columns = ['avg_shop_category_cnt_monthly']

avg_shop_category_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_shop_category_cnt_monthly, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

train_monthly['avg_shop_category_cnt_monthly'] = train_monthly['avg_shop_category_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_shop_category_cnt_monthly')
avg_shop_type_cnt_monthly = train_monthly.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})

avg_shop_type_cnt_monthly.columns = ['avg_shop_type_cnt_monthly']

avg_shop_type_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_shop_type_cnt_monthly, on=['date_block_num', 'shop_id', 'type_code'], how='left')

train_monthly['avg_shop_type_cnt_monthly'] = train_monthly['avg_shop_type_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_shop_type_cnt_monthly')
avg_shop_subtype_cnt_monthly = train_monthly.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})

avg_shop_subtype_cnt_monthly.columns = ['avg_shop_subtype_cnt_monthly']

avg_shop_subtype_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_shop_subtype_cnt_monthly, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')

train_monthly['avg_shop_subtype_cnt_monthly'] = train_monthly['avg_shop_subtype_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_shop_subtype_cnt_monthly')
avg_city_cnt_monthly = train_monthly.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})

avg_city_cnt_monthly.columns = ['avg_city_cnt_monthly']

avg_city_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_city_cnt_monthly, on=['date_block_num', 'city_code'], how='left')

train_monthly['avg_city_cnt_monthly'] = train_monthly['avg_city_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_city_cnt_monthly')
avg_city_item_cnt_monthly = train_monthly.groupby(['date_block_num', 'city_code', 'item_id']).agg({'item_cnt_month': ['mean']})

avg_city_item_cnt_monthly.columns = ['avg_city_item_cnt_monthly']

avg_city_item_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_city_item_cnt_monthly, on=['date_block_num', 'city_code', 'item_id'], how='left')

train_monthly['avg_city_item_cnt_monthly'] = train_monthly['avg_city_item_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_city_item_cnt_monthly')
avg_city_category_cnt_monthly = train_monthly.groupby(['date_block_num', 'city_code', 'item_category_id']).agg({'item_cnt_month': ['mean']})

avg_city_category_cnt_monthly.columns = ['avg_city_category_cnt_monthly']

avg_city_category_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_city_category_cnt_monthly, on=['date_block_num', 'city_code', 'item_category_id'], how='left')

train_monthly['avg_city_category_cnt_monthly'] = train_monthly['avg_city_category_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_city_category_cnt_monthly')
avg_city_type_cnt_monthly = train_monthly.groupby(['date_block_num', 'city_code', 'type_code']).agg({'item_cnt_month': ['mean']})

avg_city_type_cnt_monthly.columns = ['avg_city_type_cnt_monthly']

avg_city_type_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_city_type_cnt_monthly, on=['date_block_num', 'city_code', 'type_code'], how='left')

train_monthly['avg_city_type_cnt_monthly'] = train_monthly['avg_city_type_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_city_type_cnt_monthly')
avg_city_subtype_cnt_monthly = train_monthly.groupby(['date_block_num', 'city_code', 'subtype_code']).agg({'item_cnt_month': ['mean']})

avg_city_subtype_cnt_monthly.columns = ['avg_city_subtype_cnt_monthly']

avg_city_subtype_cnt_monthly.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, avg_city_subtype_cnt_monthly, on=['date_block_num', 'city_code', 'subtype_code'], how='left')

train_monthly['avg_city_subtype_cnt_monthly'] = train_monthly['avg_city_subtype_cnt_monthly'].astype(np.float16)

train_monthly =  traceback_feature(train_monthly, 'avg_city_subtype_cnt_monthly')
# 某商品平均价格

group = train.groupby(['item_id']).agg({'item_price': ['mean']})

group.columns = ['avg_item_price']

group.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, group, on=['item_id'], how='left')

train_monthly['avg_item_price'] = train_monthly['avg_item_price'].astype(np.float16)
group = train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})

group.columns = ['avg_item_price_monthly']

group.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, group, on=['date_block_num', 'item_id'], how='left')

train_monthly['avg_item_price_monthly'] = train_monthly['avg_item_price_monthly'].astype(np.float16)
# 过去六个月的月平均价格作为重要参考特征

train_monthly = traceback_feature(train_monthly, 'avg_item_price_monthly', previous=[1, 2, 4, 5, 6])

train_monthly = traceback_feature(train_monthly, 'avg_item_price_monthly', previous=[12])
train_monthly['delta_item_price_monthly'] =(train_monthly['avg_item_price_monthly'] - train_monthly['avg_item_price'])/train_monthly['avg_item_price']
previous_monthes = [1, 2, 4, 5, 6]

for i in previous_monthes:

    train_monthly['prev_{}_{}'.format(i, 'delta_item_price_monthly')] = (train_monthly['prev_{}_{}'.format(i, 'avg_item_price_monthly')] - train_monthly['avg_item_price'])/train_monthly['avg_item_price']
def select_trend(row, previous_monthes = [1, 2, 4, 5, 6]):

    """

    选择最近半年的最近一次的平均价格波动，作为特征

    """

    for i in previous_monthes:

        if row['prev_{}_{}'.format(i, 'delta_item_price_monthly')]:

            return row['prev_{}_{}'.format(i, 'delta_item_price_monthly')]

    return 0

    

train_monthly['delta_price_prev'] = train_monthly.apply(select_trend, axis=1)

train_monthly['delta_price_prev'] = train_monthly['delta_price_prev'].astype(np.float16)
# 过去六个月的平均价格波动用最近一次的价格波动替代，其他特征则舍弃

fetures_to_drop = []

for i in previous_monthes:

    fetures_to_drop += ['prev_{}_{}'.format(i, 'delta_item_price_monthly')]



train_monthly.drop(fetures_to_drop, axis=1, inplace=True)
group = train_monthly.groupby(['date_block_num','shop_id']).agg({'revenue_monthly': ['sum']})

group.columns = ['shop_revenue_monthly']

group.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, group, on=['date_block_num','shop_id'], how='left')

train_monthly['shop_revenue_monthly'] = train_monthly['shop_revenue_monthly'].astype(np.float32)



group = group.groupby(['shop_id']).agg({'shop_revenue_monthly': ['mean']})

group.columns = ['shop_avg_revenue']

group.reset_index(inplace=True)



train_monthly = pd.merge(train_monthly, group, on=['shop_id'], how='left')

train_monthly['shop_avg_revenue'] = train_monthly['shop_avg_revenue'].astype(np.float32)
train_monthly['delta_revenue'] = (train_monthly['shop_revenue_monthly'] - train_monthly['shop_avg_revenue']) / train_monthly['shop_avg_revenue']

train_monthly['delta_revenue'] = train_monthly['delta_revenue'].astype(np.float16)



train_monthly = traceback_feature(train_monthly, 'delta_revenue', previous=[1])



train_monthly.drop(['shop_revenue_monthly','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
# 考虑年度的季节周期性趋势

train_monthly['month'] = train_monthly['date_block_num'] % 12
# 不同月份的天数作为新特征

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

train_monthly['days'] = train_monthly['month'].map(days).astype(np.int8)
# 离某店某商品上次销售相隔的月份数

cache = {}

train_monthly['item_shop_last_sale'] = -1

train_monthly['item_shop_last_sale'] = train_monthly['item_shop_last_sale'].astype(np.int8)

for idx, row in train_monthly.iterrows():    

    key = str(row.item_id)+' '+str(row.shop_id)

    if key not in cache:

        if row.item_cnt_month != 0:

            cache[key] = row.date_block_num

    else:

        last_date_block_num = cache[key]

        train_monthly.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num

        cache[key] = row.date_block_num
# 离某商品上次销售相隔的月份数

cache = {}

train_monthly['item_last_sale'] = -1

train_monthly['item_last_sale'] = train_monthly['item_last_sale'].astype(np.int8)

for idx, row in train_monthly.iterrows():    

    key = row.item_id

    if key not in cache:

        if row.item_cnt_month != 0:

            cache[key] = row.date_block_num

    else:

        last_date_block_num = cache[key]

        if row.date_block_num>last_date_block_num:

            train_monthly.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num

            cache[key] = row.date_block_num     
# 离第一次销售所间隔的月份数

train_monthly['item_shop_first_sale'] = train_monthly['date_block_num'] - train_monthly.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

train_monthly['item_first_sale'] = train_monthly['date_block_num'] - train_monthly.groupby('item_id')['date_block_num'].transform('min')
train_monthly.date_block_num.unique()
train_monthly = train_monthly[train_monthly.date_block_num > 11]
train_monthly.date_block_num.unique()
def fill_na(df):

#     for col in df.columns:

#         if ('prev' in col) & (df[col].isnull().any()):

#             if ('item_cnt' in col):

#                 df[col].fillna(0, inplace=True)         

#     return df

    for col in df.columns:

        median = df[col].median()

        df[col].fillna(median, inplace=True)



# train_monthly = fill_na(train_monthly)

train_monthly.fillna(0, inplace=True)
# 测试集中商店的数量

print(f"shop numbers in test set: {len(set(test.shop_id))}")

# 测试集中商品数量

print(f"item numbers in test set: {len(set(test.item_id))}")

# 测试集中有而训练集中未见过的商品

new_test_items = set(test.item_id) - set(test.item_id).intersection(set(train.item_id))

print("the number of new items in test set compared to train set: ", len(new_test_items))
test.head()
# 预测 2015 年 11 月(即date_block_num=34)某商店的某样商品的价格

test['date_block_num'] = 34
# 按照数据类型压缩数据集

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)
test.info()
X_train = train_monthly[train_monthly.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = train_monthly[train_monthly.date_block_num < 33]['item_cnt_month']

X_valid = train_monthly[train_monthly.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = train_monthly[train_monthly.date_block_num == 33]['item_cnt_month']

X_test = train_monthly[train_monthly.date_block_num == 34].drop(['item_cnt_month'], axis=1)
from xgboost import XGBRegressor

from xgboost import plot_importance

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing





model = XGBRegressor(

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,  

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
# 提交文件

Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('submission.csv', index=False)
def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



plot_features(model, (10,14))