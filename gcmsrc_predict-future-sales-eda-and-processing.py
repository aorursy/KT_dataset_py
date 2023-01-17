! pip install googletrans
! pip install strsim
import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from googletrans import Translator

import re

from tqdm import tqdm_notebook

import gc

from itertools import product
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

    for f in float_cols:

        df.loc[:,f] = pd.to_numeric(df[f], downcast='float')

    

    for i in int_cols:

        df.loc[:,i] = pd.to_numeric(df[i], downcast='integer')

    

    return df
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
train.head()
test.head()
fig = plt.figure(figsize=(18,9))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((3,3), (0,0), colspan = 3)

counts = train['shop_id'].value_counts(normalize=True).sort_values(ascending=False)

sns.barplot(x = counts.index, y=counts, order=counts.index)

plt.title("Number of transactions by shop ID (normalized)")



plt.subplot2grid((3,3), (1,0))

sns.distplot(train.item_id)

plt.title("Item ID histogram")



plt.subplot2grid((3,3), (1,1))

sns.distplot(train.item_price)

plt.title("Item price histogram")



plt.subplot2grid((3,3), (1,2))

sns.distplot(train.item_cnt_day)

plt.title("Item count day histogram")



plt.subplot2grid((3,3), (2,0), colspan=3)

counts = train['date_block_num'].value_counts(normalize=True).sort_values(ascending=False)

sns.barplot(x=counts.index, y=counts, order=counts.index)

plt.title("Number of transactions per date block num");
train['item_id'].value_counts(ascending=False)[:5]
items.loc[items['item_id']==20949]
translator = Translator()

translator.translate(items.loc[items['item_id']==20949].item_name.values[0]).text
test[test.item_id==20949].head()
train.item_cnt_day.sort_values(ascending=False)[:10]
train[train.item_cnt_day>2000]
items[items.item_id==11373]
translator.translate(items[items.item_id==11373].item_name.values[0]).text
train[(train.item_id==11373)&(train.item_cnt_day<2000)]['item_cnt_day'].median()
train = train[train.item_cnt_day < 2000]
train[train.duplicated(subset=['date', 'shop_id', 'item_id'], keep=False)]
train[train.duplicated(keep=False)]
print(train.shape)

train = train[~train.duplicated()]

print(train.shape)
train.item_price.sort_values(ascending=False)[:10]
train[train.item_price>300000]
items[items.item_id==6066]
translator.translate(items[items.item_id==6066].item_name.values[0]).text
train[train.item_id==6066]
print(train.shape)

train = train[train.item_price<300000]

print(train.shape)
train[train.item_price <= 0]
train[train.item_id==2973].head()
median_price_item_2973 = train[(train.item_id==2973)&(train.date_block_num==4)&(train.shop_id==32)&(train.item_price>0)]['item_price'].median()

train.loc[484683,'item_price'] = median_price_item_2973
fig = plt.figure(figsize=(18,9))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((3,3), (0,0), colspan = 3)

counts = train['shop_id'].value_counts(normalize=True).sort_values(ascending=False)

sns.barplot(x = counts.index, y=counts, order=counts.index)

plt.title("Number of transactions by shop ID (normalized)")



plt.subplot2grid((3,3), (1,0))

sns.distplot(train.item_id)

plt.title("Item ID histogram")



plt.subplot2grid((3,3), (1,1))

sns.distplot(train.item_price)

plt.title("Item price histogram")



plt.subplot2grid((3,3), (1,2))

sns.distplot(train.item_cnt_day)

plt.title("Item count day histogram")



plt.subplot2grid((3,3), (2,0), colspan=3)

counts = train['date_block_num'].value_counts(normalize=True).sort_values(ascending=False)

sns.barplot(x=counts.index, y=counts, order=counts.index)

plt.title("Number of transactions per date block num");
fig = plt.figure(figsize=(18,9))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((3,3), (0,0), colspan = 3)

counts = test['shop_id'].value_counts(normalize=True).sort_values(ascending=False)

sns.barplot(x = counts.index, y=counts, order=counts.index)

plt.title("Number of transactions by shop ID (normalized)")



plt.subplot2grid((3,3), (1,0))

sns.distplot(test.item_id)

plt.title("Item ID histogram");
print("Number of item in test set: {}".format(len(test.item_id.unique())))

item_not_in_train_set = test[~test.item_id.isin(train.item_id.unique())].item_id.sort_values().unique()

print("Number of item in test set with no transaction in train: {}".format(len(item_not_in_train_set)))
item_not_in_train_set[:10]
items[items.item_id.isin(range(198, 210))]
items.loc[204,'item_name']
del counts

del item_not_in_train_set

gc.collect()
shops_not_in_train = test[~test.shop_id.isin(train.shop_id.unique())].shop_id.unique()

print("Number of shops in test with no transaction in train: {}".format(len(shops_not_in_train)))
shops.shop_name[:5]
shop_splitter = re.compile(r'(\w+)\s(.*)')

shop_names = shops.shop_name.apply(lambda x: shop_splitter.search(x).groups())

shop_names_df = pd.DataFrame(shop_names.values.tolist(), columns=['city', 'extracted_name'])

shop_names_df.head()
shops = pd.concat([shops, shop_names_df], axis=1)

shops.head()
shops.loc[:,'is_city'] = shops.city.apply(lambda x :0 if x in ['Выездная', 'магазин', 'Цифровой'] else 1)
shops.shop_name.unique()
def shop_sub_type(x):

    if x[0] == 0:

        return 'non_city'

    else:

        if 'ТЦ' in x[1]:

            return 'ТЦ'

        elif 'ТРЦ' in x[1]:

            return 'ТРЦ'

        elif 'ТРК' in x[1]:

            return 'ТРК'

        elif 'Орджоникидзе' in x[1]:

            return 'Орджоникидзе'

        else:

            return 'other'
shops.loc[:,'shop_sub_type'] = shops[['is_city', 'extracted_name']].apply(shop_sub_type, axis=1)

shops.head()
shops[shops.shop_name.duplicated(keep=False)]
del shop_names_df, shops_not_in_train

gc.collect()
from similarity.normalized_levenshtein import NormalizedLevenshtein
unique_shop_id = shops.shop_id.unique()

similarity_grid = np.zeros(shape=(len(unique_shop_id), len(unique_shop_id)))
norm_lev = NormalizedLevenshtein()



for i in unique_shop_id:

    for j in unique_shop_id:

        distance = norm_lev.similarity(shops[shops.shop_id==i].shop_name.values[0], shops[shops.shop_id==j].shop_name.values[0])

        similarity_grid[i,j] = distance
fig, ax = plt.subplots(figsize=(10,8))

mask = similarity_grid < 0.6

sns.heatmap(similarity_grid, ax=ax, mask=mask, cmap = sns.color_palette('Blues'))

ax.set_facecolor("grey")
indices = zip(*np.triu_indices_from(similarity_grid))
similar_stores = []



for c in indices:

    i, j = c[0], c[1]

    if i != j and similarity_grid[i,j]>0.6:

        similar_stores.append([i,j, similarity_grid[i,j]])

similar_stores = pd.DataFrame(similar_stores, columns=['i','j','similarity'])

similar_stores.sort_values(by='similarity',ascending=False, inplace=True)

similar_stores
shops[shops.shop_id.isin([10,11])].shop_name
train.loc[train.shop_id==10, 'shop_id'] = 11

test.loc[test.shop_id==10, 'shop_id'] = 11
shops[shops.shop_id.isin([23,24])].shop_name
shops[shops.shop_id.isin([30,31])].shop_name
shops[shops.shop_id.isin([0,57])].shop_name
train.loc[train.shop_id==57, 'shop_id'] = 0

test.loc[test.shop_id==57, 'shop_id'] = 0
shops[shops.shop_id.isin([1,58])].shop_name
train.loc[train.shop_id==58, 'shop_id'] = 1

test.loc[test.shop_id==58, 'shop_id'] = 1
shops[shops.shop_id.isin([39,40])].shop_name
shops[shops.shop_id.isin([38,54])].shop_name
del similar_stores, similarity_grid

gc.collect()
categories.item_category_name.head()
split_names = categories.item_category_name.apply(lambda x: [x.strip() for x in x.split(' - ')])
new_categories = np.chararray((len(categories), 2), itemsize=33, unicode=True)

new_categories[:] = 'None'



# Add categories with a for loop

for i, c_list in enumerate(split_names):

    for j, c_value in enumerate(c_list):

        new_categories[i,j] = c_value
new_categories_df = pd.DataFrame(new_categories, columns=['category', 'sub_category'])

categories = categories.join(new_categories_df)



# If sub_category is None replace it with category

categories.loc[:,'sub_category'] = categories[['category', 'sub_category']].apply(lambda x: x[0] if x[1]=='None' else x[1], axis=1)



categories.head()
del split_names, new_categories, new_categories_df

gc.collect()
items[items.item_name.str.contains('FIFA 13')]
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
# Please notice I had to change the default token_pattern to also tokenize 1-character word

# (e.g. Far Cry 3 and Far Cry 2)

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w*\\b')

vectorized_names = vectorizer.fit_transform(items.item_name.values)
# Calculate cosine similarity grid

cosine_similarity_grid = cosine_similarity(vectorized_names)
# Let's print out the most similar names (excluding same names)

indices = zip(*np.triu_indices_from(cosine_similarity_grid))

similar_items = []



for c in tqdm_notebook(indices):

    i, j = c[0], c[1]

    if i != j and cosine_similarity_grid[i,j]>0.9:

        similar_items.append([i,j, cosine_similarity_grid[i,j]])

similar_items = pd.DataFrame(similar_items, columns=['i','j','similarity'])

similar_items.sort_values(by='similarity',ascending=False, inplace=True)
similar_items[similar_items.similarity==1].shape
similar_items[similar_items.similarity==1].tail()
items[items.item_id.isin([8642, 8643, 8632, 8633])]
items[items.item_id.isin([9048, 9049, 18126, 18127])]
duplicated_items = similar_items[similar_items.similarity==1].copy()

for c in duplicated_items.columns:

    duplicated_items.loc[:,c] = pd.to_numeric(duplicated_items[c], downcast='integer')
for _, r in tqdm_notebook(duplicated_items.iterrows()):

    train.loc[train.item_id==r[1], 'item_id'] = r[0]

    test.loc[test.item_id==r[1], 'item_id'] = r[0]
train[train.item_id==9049]
del duplicated_items

gc.collect()
similar_items[(similar_items.similarity>0.99)&(similar_items.similarity<1)]
items[items.item_id.isin([4199, 4200, 10479, 10480, 14431, 14432])]
# Take a sample

sample_idx = np.random.choice(np.arange(22170), size=250, replace=False)

sample = cosine_similarity_grid[sample_idx].copy()
from sklearn.manifold import TSNE

sim_embed = TSNE().fit_transform(sample)

x, y = zip(*sim_embed)

plt.scatter(x,y);
del sim_embed, sample

gc.collect()
from torch import topk

import torch
cosine_similarity_grid_torch = torch.from_numpy(cosine_similarity_grid)
topk_values, topk_indices = topk(cosine_similarity_grid_torch, 4)

topk_values, topk_indices = topk_values.numpy(), topk_indices.numpy()
def add_index(i, n, topk_values, topk_indices, threshold=0.6):

    

    val, ind = topk_values[i], topk_indices[i]

    

    if val[n] > threshold:

        return ind[n]

    else:

        return i
# Create similar 1 column

items.loc[:,'similar_1'] = items.item_id.apply(add_index, n=1, topk_values=topk_values, topk_indices=topk_indices)
def add_index_mul(i, n, topk_values, topk_indices, threshold=0.6):

    

    item_id, similar_prev = i[0], i[1]

    

    val, ind = topk_values[item_id], topk_indices[item_id]

    

    if val[n] > threshold:

        return ind[n]

    else:

        return similar_prev
# Create similar 2 columns

items.loc[:,'similar_2'] = items[['item_id', 'similar_1']].apply(add_index_mul, n=2, topk_values=topk_values, topk_indices=topk_indices, axis=1)



# Create similar 3 columns

items.loc[:,'similar_3'] = items[['item_id', 'similar_2']].apply(add_index_mul, n=3, topk_values=topk_values, topk_indices=topk_indices, axis=1)
# Let's check out the FIFA example again

items[items.item_name.str.contains('FIFA 14')]
del similar_items, cosine_similarity_grid, topk_values, topk_indices, cosine_similarity_grid_torch

del vectorized_names

gc.collect()
items = items.drop('item_name', axis=1)
def frequency_encode(series):

    return series.value_counts(normalize=True)
#Add shop_id and item_id combinationa

train.loc[:,'shop_and_item'] = train.shop_id.astype(str) + '-' + train.item_id.astype(str)

train.head()
from sklearn.preprocessing import LabelEncoder
# Create all possible shop and item combinations

shop_and_item = pd.Series(list(product(shops.shop_id.values, items.item_id.values)))

shop_and_item = pd.DataFrame(shop_and_item.apply(lambda x: str(x[0]) + '-' + str(x[1])), columns=['shop_and_item'])



# Label-encode them

shop_and_item_encoder = LabelEncoder()

shop_and_item_encoder.fit(shop_and_item.shop_and_item)



# Transform

shop_and_item.loc[:,'shop_and_item'] = shop_and_item_encoder.transform(shop_and_item.shop_and_item)

train.loc[:,'shop_and_item'] = shop_and_item_encoder.transform(train.shop_and_item)
# Create frequency encodings in items dataframe

items.loc[:,'item_id_freq_encod'] = items.item_id.map(frequency_encode(train.item_id))
# Frequency encode shop_id

shops.loc[:,'shop_id_freq_encod'] = shops.shop_id.map(frequency_encode(train.shop_id))
# Add shops details

train = train.merge(shops[['shop_id', 'city', 'shop_sub_type']], how='left', on=['shop_id'])



# Add category id

train = train.merge(items[['item_id', 'item_category_id']], how='left', on=['item_id'])



# Add category information

train = train.merge(categories[['item_category_id', 'category', 'sub_category']], how='left', on=['item_category_id'])
# Add city freq encoding

shops.loc[:,'city_freq_encod'] = shops.city.map(frequency_encode(shops.city))



# Add category_id freq encoding

categories.loc[:,'item_category_id_freq_encod'] = categories.item_category_id.map(frequency_encode(train.item_category_id))



# Add category freq encoding

categories.loc[:,'category_freq_encod'] = categories.category.map(frequency_encode(train.category))



# Add sub_category freq encoding

categories.loc[:,'sub_category_freq_encod'] = categories.sub_category.map(frequency_encode(train.sub_category))



# Add shop_item freq encoding

shop_and_item.loc[:,'shop_and_item_freq_encod'] = shop_and_item.shop_and_item.map(frequency_encode(train.shop_and_item))
# Fill na

items = items.fillna(0)

categories = categories.fillna(0)

shops = shops.fillna(0)

shop_and_item = shop_and_item.fillna(0)
# Add oldest transaction (don't have to fill NA here yet)

items.loc[:,'oldest_date_block_num'] = items.item_id.map(train.groupby('item_id')['date_block_num'].min())
# Dump modified files

items.to_hdf('processed_items.hdf5', key='df')

categories.to_hdf('processed_categories.hdf5', key='df')

shops.to_hdf('processed_shops.hdf5', key='df')

shop_and_item.to_hdf('shop_and_item.hdf5', key='df')

train.to_hdf('processed_train.hdf5', key='df')

test.to_hdf('processed_test.hdf5', key='df')
grid = []

for block_num in tqdm_notebook(train.date_block_num.unique()):

    cur_shops = train[train['date_block_num']==block_num]['shop_id'].unique()

    cur_items = train[train['date_block_num']==block_num]['item_id'].unique()

    grid.append(np.array(list(product(cur_shops, cur_items, [block_num])), dtype='int32'))
# Create dataframe from grid

index_cols = ['shop_id', 'item_id', 'date_block_num']

grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)

grid.sort_values(by=['date_block_num', 'shop_id', 'item_id'], inplace=True)

grid.reset_index(inplace=True, drop=True)
grid.head()
# Add item_cnt_month (not target)

item_cnt_df = train.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

item_cnt_df.head()
# Clip values

item_cnt_df.loc[:,'item_cnt_month'] = item_cnt_df.item_cnt_month.clip(0,20)
# Merge item_cnt_month into grid (NaN values fill them with 0)

grid = grid.merge(item_cnt_df, how='left', on=['date_block_num', 'shop_id', 'item_id']).fillna(0)
del item_cnt_df

gc.collect()
grid.head()
test.loc[:,'date_block_num'] = 34

test.loc[:,'item_cnt_month'] = 0
grid = grid.append(test.drop('ID', axis=1))

grid.loc[:,'item_cnt_month'] = grid.item_cnt_month.astype(int)
# Add shop and item

grid.loc[:,'shop_and_item'] = grid.shop_id.astype(str) + '-' + grid.item_id.astype(str)

grid.loc[:,'shop_and_item'] = shop_and_item_encoder.transform(grid.shop_and_item)
grid.head()
del train

del test

gc.collect()
def generate_lag(grid, months, lag_column):

    for month in months:

        # Speed up by grabbing only the useful bits

        

        grid_shift = grid[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()

        grid_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]

        grid_shift['date_block_num'] += month

        grid = pd.merge(grid, grid_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    return grid
grid = downcast_dtypes(grid)
# Lag item counts

%time

grid = generate_lag(grid, [1,2,3,4,5,6,11,12], 'item_cnt_month')
# Fill na with zero (later remember to select only date_block_num greater or equal to 12)

grid = grid.fillna(0)
for c in grid.columns[5:]:

    grid.loc[:,c] = pd.to_numeric(grid[c], downcast='integer')
grid = downcast_dtypes(grid)
from sklearn.preprocessing import LabelEncoder
# Fix shops

encoder = LabelEncoder()

shops.loc[:,'city'] = encoder.fit_transform(shops.city)

shops.loc[:,'shop_sub_type'] = encoder.fit_transform(shops.shop_sub_type)
# Fix categories

encoder = LabelEncoder()

categories.loc[:,'category'] = encoder.fit_transform(categories.category)

categories.loc[:,'sub_category'] = encoder.fit_transform(categories.sub_category)
# Drop some columns

categories.drop(columns=['item_category_name'], inplace=True)

shops.drop(columns=['shop_name', 'extracted_name'], inplace=True)
# Add all to grid

grid = grid.merge(items, how='left', on=['item_id'])

del items

gc.collect()

grid = downcast_dtypes(grid)
grid = grid.merge(categories, how='left', on=['item_category_id'])

del categories

gc.collect()

grid = downcast_dtypes(grid)
grid = grid.merge(shops, how='left', on=['shop_id'])

del shops

gc.collect()

grid = downcast_dtypes(grid)
grid = grid.merge(shop_and_item, how='left', on=['shop_and_item'])

del shop_and_item

gc.collect()

grid.drop(columns=['shop_and_item'])

grid = downcast_dtypes(grid)
grid = downcast_dtypes(grid)
del shop_names, shop_splitter, x, y

gc.collect()
grid.to_hdf('grid.hdf5', key='df')
# # Mean item_id

# mean_id = grid.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()

# grid = grid.merge(mean_id, how='left', on=['date_block_num', 'item_id'])



# # Delete mean_id

# del mean_id

# gc.collect()



# # Create lags

# grid = generate_lag(grid, [1,2,3,4,5,6,11,12], 'item_month_mean')



# # We need to drop item_month_mean otherwise that would be a massive leakage for our model

# # Item month mean is basically the average target value for the product.

# grid.drop(columns=['item_month_mean'], inplace=True)
# grid = downcast_dtypes(grid)
# # Mean shop_id (should capture the activity of a shop)

# mean_shop_id = grid.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()

# grid = grid.merge(mean_shop_id, how='left', on=['date_block_num', 'shop_id'])



# # Delete mean_id

# del mean_shop_id

# gc.collect()



# # Create lags

# grid = generate_lag(grid, [1,2,3,4,5,6,11,12], 'shop_month_mean')



# # We need to drop shop_month_mean otherwise that would be a massive leakage for our model

# grid.drop(columns=['shop_month_mean'], inplace=True)
# grid = downcast_dtypes(grid)
# # Mean city

# mean_city_id = grid.groupby(['date_block_num', 'city'])['item_cnt_month'].mean().rename('city_month_mean').reset_index()

# grid = grid.merge(mean_city_id, how='left', on=['date_block_num', 'city'])



# # Delete mean_id

# del mean_city_id

# gc.collect()



# # Create lags

# grid = generate_lag(grid, [1,2,3,4,5,6,11,12], 'city_month_mean')



# # We need to drop city_month_mean otherwise that would be a massive leakage for our model

# grid.drop(columns=['city_month_mean'], inplace=True)
# grid = downcast_dtypes(grid)
# # Mean category_id

# mean_category_id = grid.groupby(['date_block_num', 'item_category_id'])['item_cnt_month'].mean().rename('category_id_month_mean').reset_index()

# grid = grid.merge(mean_category_id, how='left', on=['date_block_num', 'item_category_id'])



# # Delete mean_id

# del mean_category_id

# gc.collect()



# # Create lags

# grid = generate_lag(grid, [1,2,3,4,5,6,11,12], 'category_id_month_mean')



# # We need to drop category_id_month_mean otherwise that would be a massive leakage for our model

# grid.drop(columns=['category_id_month_mean'], inplace=True)
# grid = downcast_dtypes(grid)
# # Mean category

# # Mean category_id

# mean_category = grid.groupby(['date_block_num', 'category'])['item_cnt_month'].mean().rename('category_month_mean').reset_index()

# grid = grid.merge(mean_category, how='left', on=['date_block_num', 'category'])



# # Delete mean_id

# del mean_category

# gc.collect()



# # Create lags

# grid = generate_lag(grid, [1,2,3,4,5,6,11,12], 'category_month_mean')



# # We need to drop category_id_month_mean otherwise that would be a massive leakage for our model

# grid.drop(columns=['category_month_mean'], inplace=True)
# grid = downcast_dtypes(grid)
# Mean sub-category