# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/sales_train.csv.gz')
item_desc = pd.read_csv('../input/items.csv', low_memory=True)
test = pd.read_csv('../input/test.csv.gz', low_memory=True)
shops = pd.read_csv('../input/shops.csv')
item_cat = pd.read_csv('../input/item_categories.csv', low_memory=True)
train.head()
#Remove the duplicate rows
print('Before drop train shape:', train.shape) 
train.drop_duplicates(subset=['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'], keep='first', inplace=True) 
train.reset_index(drop=True, inplace=True) 
print('After drop train shape:', train.shape)
train.describe()
from matplotlib import pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(ncols = 2, figsize = (20, 10))
sns.boxplot(x='item_price', data = train, ax = ax[0])
ax[0].set_title('Boxplot for Item price')
sns.boxplot(x='item_cnt_day', data = train, ax = ax[1])
ax[1].set_title('Boxplot for Item counts')
plt.show()
train = train.loc[(train.item_price < 100000) & (train.item_cnt_day <= 1000) & (train.item_price >=0)]
# Perform the aggregation according to every month.

dfs = []
for month_block in train.date_block_num.unique():
    print('Making dataframe for: %s' %(month_block+1))
    df = pd.DataFrame(train[train['date_block_num'] == month_block].groupby(['shop_id', 'item_id'])['item_cnt_day'].sum())
    df.reset_index(inplace = True)
    df['date_block_num'] = month_block
    dfs.append(df)
    
df = pd.DataFrame()
for frame in dfs:
    df = pd.concat([df, frame], axis = 0)
train = df.copy()
!pip install missingno #check this package out for useful exploration of missing data in dataset.
import missingno as msno

def aggregate_cols(row):
    return str(int(row['shop_id'])) + str(int(row['item_id']))
train['agg_id'] = train.apply(lambda row: aggregate_cols(row), axis = 1)
test['agg_id'] = test.apply(lambda row: aggregate_cols(row), axis = 1)

dic = {}
ids = train.agg_id.unique().tolist()
for id  in ids:
    dic[id] = 0

test['agg_id'] = test['agg_id'].map(dic)

plt.style.use('ggplot')
msno.bar(test)
train = pd.merge(train, shops, on = 'shop_id', how = 'inner')
Z = dict(train['shop_name'].value_counts())
fig, ax = plt.subplots(1, figsize=(15, 5))
sns.stripplot(list(Z.keys()), list(Z.values()), ax = ax)
plt.xticks(rotation = 90)
plt.show()
item_desc = pd.merge(item_desc, item_cat, how='inner', on='item_category_id')
train = pd.merge(train, item_desc[['item_id', 'item_category_name', 'item_category_id']], on = 'item_id', how = 'inner')

Z = dict(train['item_category_name'].value_counts())
fig, ax = plt.subplots(1, figsize=(18, 5))
sns.stripplot(list(Z.keys()), list(Z.values()), ax = ax, edgecolor='black', size=5)
plt.xticks(rotation = 90)
plt.title('Item Categories set according to Frequency')
plt.show()

sns.jointplot('shop_id', 'item_category_id', data = train, space = 0, size = 15, ratio = 5)
plt.yticks(range(90))
plt.show()
test = pd.merge(test, item_desc[['item_id', 'item_category_name', 'item_category_id']], on = 'item_id', how = 'inner')
sns.jointplot('shop_id', 'item_category_id', data = test, space = 0, size = 15, ratio = 5)
plt.yticks(range(90))
plt.show()
test['date_block_num'] = 34
g = pd.concat([train[['shop_id', 'date_block_num']], test[['shop_id', 'date_block_num']]])
sns.jointplot('date_block_num', 'shop_id', data = g, space = 0, size = 15, ratio = 5)
plt.yticks(range(90))
plt.show()
df = train[train.date_block_num >= 12]
months = np.sort(df.date_block_num.unique())
new_items_introduced = [0] * 12
for i in range(12, len(months)):
    new_items_introduced[i%12] += len(np.setdiff1d(df[df.date_block_num == months[i]]['item_id'].unique(), df[df.date_block_num < months[i]]['item_id'].unique()))

#names = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']       
plt.figure(figsize = (15, 5))
plt.bar(np.arange(1,13), new_items_introduced)
plt.title('Items introduced over months')
plt.xticks(np.arange(1,13))
plt.show()
'''
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(new_items_introduced, labels=names)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
'''
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
hb1 = ax.hexbin(train.shop_id, train.item_id, cmap = 'inferno')
plt.title('Hexagonal Binning for the items present in the shops')
plt.xticks(np.arange(0,60))
plt.xlabel('Shop Id')
plt.ylabel('Item Id')
plt.show()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
hb1 = ax.hexbin(item_desc.item_category_id, item_desc.item_id, cmap = 'ocean')
plt.title('Hexagonal Binning for the item categories vs items')
plt.xticks(np.arange(0,85))
plt.xlabel('Item Category Id')
plt.ylabel('Item Id')
plt.show()
Z = train.groupby('date_block_num').agg({'item_cnt_day': sum}).reset_index()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize = (20,10))
sns.barplot(data=Z, x='date_block_num', y='item_cnt_day', ax = ax, palette="BrBG")
plt.title('Overall Item counts over the course of 3 years')
plt.show()