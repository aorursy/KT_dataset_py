import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import pickle

from itertools import product

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

shop = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_category = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
print(test.describe())

print(test.head(5))

print(shop.describe())

print(shop.head(5))

print(sales.describe())

print(sales.head(5))

print(items.describe())

print(items.head(5))

print(item_category.describe())

print(item_category.head(5))

print(sample_submission.head(5))
print(sales.shop_id.nunique() == shop.shop_id.size)



print(sales.loc[sales.item_price < 0])



print(sales.loc[(sales.item_id == 2973) & (sales.shop_id == 32)])



median = sales[(sales.shop_id==32)&(sales.item_id==2973)&(sales.date_block_num==4)&(sales.item_price>0)].item_price.median()

sales.loc[sales.item_price<0, 'item_price'] = median



plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=sales.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(sales.item_price.min(), sales.item_price.max())

sns.boxplot(x=sales.item_price)
sales = sales[sales.item_cnt_day<1000]
# print(sales.shop_id.nunique() > test.shop_id.nunique())



# sales_shop_item_pair = set([(i.shop_id,i.item_id) for i in sales.itertuples()])

# test_shop_item_pair = set([(i.shop_id,i.item_id) for i in test.itertuples()])



# sales_shop = set([i.shop_id for i in sales.itertuples()])

# test_shop = set([i.shop_id for i in test.itertuples()])



# sales_item = set([i.item_id for i in sales.itertuples()])

# test_item = set([i.item_id for i in test.itertuples()])



# print('Test shop,item pair not in sales',len(test_shop_item_pair.difference(sales_shop_item_pair)))

# print('Test shop not in sales', len(test_shop.difference(sales_shop)))

# print('Test item not in sales', len(test_item.difference(sales_item)))





index_cols = ['shop_id', 'item_id', 'date_block_num']



# For every month we create a grid from all shops/items combinations from that month

train = [] 

for block_num in sales['date_block_num'].unique():

    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()

    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()

    train.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



train = pd.DataFrame(np.vstack(train), columns = index_cols,dtype=np.int32)

test['date_block_num'] = 34

refined_data = pd.concat([train, test], ignore_index=True, sort=False)

refined_data.fillna(0, inplace=True) 





sales = pd.merge(sales, items, on=['item_id'], how='left')

refined_data = pd.merge(refined_data, items, on=['item_id'], how='left')

sales = pd.merge(sales, item_category, on=['item_category_id'], how='left')

refined_data = pd.merge(refined_data, item_category, on=['item_category_id'], how='left')

sales = pd.merge(sales, shop, on=['shop_id'], how='left')

refined_data = pd.merge(refined_data, shop, on=['shop_id'], how='left')



sales['formated_date'] = pd.to_datetime(sales['date'],infer_datetime_format=True)

sales['month'] = sales['date_block_num'] % 12

refined_data['month'] = refined_data['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

sales['days'] = sales['month'].map(days).astype(np.int8)

refined_data['days'] = refined_data['month'].map(days).astype(np.int8)



excluded_for_last_block = []







# refined_data['combined_text'] = refined_data.shop_name + ' ' + refined_data.item_category_name + ' ' + refined_data.item_name + ' ' + refined_data.date_block_num.astype(str) + ' ' + refined_data.month.astype(str) 

# vectorizer = TfidfVectorizer()



# text_data = vectorizer.fit_transform(refined_data['combined_text'])

# svd = TruncatedSVD(n_components=4)

# text_data_reduced = svd.fit_transform(text_data)



# text_features = pd.DataFrame({'c1':text_data_reduced[:,0],

#                               'c2':text_data_reduced[:,1],

#                               'c3':text_data_reduced[:,2],

#                               'c4':text_data_reduced[:,3],

#                               'date_block_num': refined_data.date_block_num,

#                        })



group = sales.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum','mean','size']})

group.columns = ['item_cnt_month','average_daily','no_of_t']



group.reset_index(inplace=True)

sales = pd.merge(sales, group, on=['date_block_num','shop_id','item_id'], how='left')

refined_data = pd.merge(refined_data, group, on=['date_block_num','shop_id','item_id'], how='left')

refined_data['item_cnt_month'] = (refined_data['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16))

refined_data['average_daily'] = (refined_data['average_daily'].fillna(0).clip(0,20).astype(np.float16))

refined_data['no_of_t'] = (refined_data['no_of_t'].fillna(0).astype(np.int8))

excluded_for_last_block.extend(['average_daily','no_of_t']) #item_cnt_month will be handled later

group = sales.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_id': ['size']})

group.columns = ['item_category_cnt']

group.reset_index(inplace=True)

sales = pd.merge(sales, group, on=['date_block_num','shop_id','item_category_id'], how='left')

refined_data = pd.merge(refined_data, group, on=['date_block_num','shop_id','item_category_id'], how='left')

excluded_for_last_block.extend(['item_category_cnt'])

group = sales.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_price': ['mean','nunique']})

group.columns = ['avg_category_price','category_price_change_count']

group.reset_index(inplace=True)

refined_data = pd.merge(refined_data, group, on=['date_block_num','shop_id','item_category_id'], how='left')

excluded_for_last_block.extend(['avg_category_price','category_price_change_count'])

group = refined_data.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['avg_category_item_cnt_month']

group.reset_index(inplace=True)

refined_data = pd.merge(refined_data, group, on=['date_block_num','shop_id','item_category_id'], how='left')

excluded_for_last_block.extend(['avg_category_item_cnt_month'])
# date_related_features = []
# sortedSale = sales.sort_values(['shop_id','item_id','formated_date'], ascending=[True, True, True])

# sortedSale['prev_date'] = sortedSale['formated_date'].shift(1)

# sortedSale['date_diff'] = (sortedSale['formated_date'] - sortedSale['prev_date'])

# sortedSale = sortedSale.loc[sortedSale.groupby(['shop_id','item_id']).cumcount() !=0]

# sortedSale['date_diff'] = sortedSale['date_diff'].dt.days.astype(np.int16)

# group = sortedSale.groupby(['shop_id','item_id']).agg({'date_diff': ['mean']})

# group.columns = ['shop_item_date_freq']

# group.reset_index(inplace=True)

# refined_data = pd.merge(refined_data, group, on=['shop_id','item_id'], how='left')

# date_related_features.extend(['shop_item_date_freq'])
# sortedSale = sales.sort_values(['shop_id','formated_date'], ascending=[True, True])

# sortedSale['prev_date'] = sortedSale['formated_date'].shift(1)

# sortedSale['date_diff'] = (sortedSale['formated_date'] - sortedSale['prev_date'])

# sortedSale = sortedSale.loc[sortedSale.groupby(['shop_id']).cumcount() !=0]

# sortedSale['date_diff'] = sortedSale['date_diff'].dt.days.astype(np.int16)

# group = sortedSale.groupby(['shop_id']).agg({'date_diff': ['mean']})

# group.columns = ['shop_date_freq']

# group.reset_index(inplace=True)

# refined_data = pd.merge(refined_data, group, on=['shop_id'], how='left')

# date_related_features.extend(['shop_date_freq'])
# sortedSale = sales.sort_values(['item_id','formated_date'], ascending=[True, True])

# sortedSale['prev_date'] = sortedSale['formated_date'].shift(1)

# sortedSale['date_diff'] = (sortedSale['formated_date'] - sortedSale['prev_date'])

# sortedSale = sortedSale.loc[sortedSale.groupby(['item_id']).cumcount() !=0]

# sortedSale['date_diff'] = sortedSale['date_diff'].dt.days.astype(np.int16)

# group = sortedSale.groupby(['item_id']).agg({'date_diff': ['mean']})

# group.columns = ['item_date_freq']

# group.reset_index(inplace=True)

# refined_data = pd.merge(refined_data, group, on=['item_id'], how='left')

# date_related_features.extend(['item_date_freq'])
# sortedSale = sales.sort_values(['item_category_id','formated_date'], ascending=[True, True])

# sortedSale['prev_date'] = sortedSale['formated_date'].shift(1)

# sortedSale['date_diff'] = (sortedSale['formated_date'] - sortedSale['prev_date'])

# sortedSale = sortedSale.loc[sortedSale.groupby(['item_category_id']).cumcount() !=0]

# sortedSale['date_diff'] = sortedSale['date_diff'].dt.days.astype(np.int16)

# group = sortedSale.groupby(['item_category_id']).agg({'date_diff': ['mean']})

# group.columns = ['cat_date_freq']

# group.reset_index(inplace=True)

# refined_data = pd.merge(refined_data, group, on=['item_category_id'], how='left')

# date_related_features.extend(['cat_date_freq'])
col = ['item_cnt_month'] + excluded_for_last_block

for lag in [1,2,3,6,12]:

    df = refined_data.copy()

    df.date_block_num+=lag

    df = df[['date_block_num','shop_id','item_id']+col]

    df.columns = ['date_block_num','shop_id','item_id'] + [var +'_lag_'+str(lag) for var in col]

    df = df[df.date_block_num < 35]

    

    refined_data = pd.merge(refined_data, df, on=['date_block_num','shop_id','item_id'], how='left')

# alpha = 100

# globalmean = 0.3343



# for category in ['item_id','shop_id','item_category_id']:

#     data = refined_data.copy()

#     data.item_cnt_month.fillna(globalmean, inplace=True) 

#     nrows = data.groupby(category).size()

#     means = data.groupby(category).item_cnt_month.agg('mean')



#     score = (np.multiply(means,nrows)  + globalmean*alpha) / (nrows+alpha)

#     data['enc_'+category] = data[category]

#     data['enc_'+category] = data['enc_'+category].map(score)

    

#     encoded_feature = data['enc_'+category].values

    

#     refined_data['enc_'+category] = encoded_feature

    
# lag_data_plus_encoding = refined_data.drop(['item_category_id','item_name','item_id','shop_id','shop_name','item_category_name','combined_text'] + excluded_for_last_block + date_related_features, axis=1)

# lag_data_plus_encoding.item_cnt_month.fillna(0,inplace=True) 

# lag_data_plus_encoding.to_pickle('lagdataandenc.pkl')

lag_only = refined_data[[col for col in refined_data.columns if ('_lag_' in col)]+['item_cnt_month','date_block_num']].fillna(0)

lag_only.to_pickle('lagonly.pkl')

# datetime_data = refined_data[date_related_features]

# datetime_data.loc[:,'shop_id'] = refined_data.copy().loc[:,'enc_shop_id']

# datetime_data.loc[:,'item_id'] = refined_data.copy().loc[:,'enc_item_id']

# datetime_data.loc[:,'date_block_num'] = refined_data.copy().loc[:,'date_block_num']

# datetime_data.loc[:,'item_cnt_month'] = refined_data.copy().loc[:,'item_cnt_month']

# datetime_data.to_pickle('datetimedata.pkl')

# text_features.loc[:,'item_cnt_month'] = refined_data.copy().loc[:,'item_cnt_month']

# text_features.to_pickle('textdata.pkl')
# lag_data_plus_encoding.dtypes
# datetime_data.dtypes
# text_features.dtypes
# lag_only.dtypes
# lag_only.to_pickle('lagonly.pkl')