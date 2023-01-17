# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
def EDA(data):

    print("-----------Top-5-Record---------")

    print(data.head(5))

    print("-----------Information---------")

    print(data.info())

    print("-----------Data types---------")

    print(data.dtypes)

    print("-----------Missingvalue---------")

    print(data.isnull().sum())

    print("-----------Null value---------")

    print(data.isna().sum())

    print("-----------shape of data---------")

    print(data.shape)

          

def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize = (16,16), bins=50, xlabelsize = 8, ylabelsize = 8)



def drop_duplicate(data, subset):

    print('Before drop shape:', data.shape)

    before = data.shape[0]

    data.drop_duplicates(subset, keep ='first', inplace = True)

    # subset is list of columns for duplicate check

    data.reset_index(drop = True, inplace = True)

    print('After drop shape:', data.shape)

    after= data.shape[0]

    print('Total Duplicate:', before-after)
EDA(sales)
sales.columns
graph_insight(sales)
# Drop duplicate data

subset = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']

drop_duplicate(sales, subset =subset)
EDA(test)

graph_insight(test)
EDA(items)

graph_insight(items)
EDA(item_cat)

graph_insight(item_cat)
def unreasonable_data(data):

    print("min value:", data.min())

    print('max value:', data.max())

    print('average value: ', data.mean())

    print('center point of data:', data.median())
unreasonable_data(sales)
print(sales['item_price'].mean())

print(sales['item_price'].std())
# -1 and 307980 looks like outliers, let's delete them

print('before sales shape:', sales.shape)

sales = sales[(sales.item_price > 0) & (sales.item_price < 300000)]

print('after sales shape:', sales.shape )
sales.info()
sales.head()
plt.figure(figsize=(20,4))

sales.groupby('date_block_num').sum()['item_cnt_day'].plot()
sales.groupby('date_block_num').sum()['item_cnt_day'].hist(figsize = (20, 4))

plt.title('sales per month histogram')

plt.xlabel('Price')

plt.figure(figsize = (20,4))

sns.lineplot(x =sales.date_block_num.unique() , y = sales.groupby('date_block_num').sum()['item_cnt_day'])

plt.title('sales per month')

plt.xlabel('Price')
unreasonable_data(sales['item_price'])
count_price = sales.item_price.value_counts().sort_index(ascending = False)

plt.subplot(221)

count_price.hist(figsize = (20,6))

plt.xlabel('Item Price', fontsize =20)

plt.title('Original Distribution')



plt.subplot(222)

sales.item_price.map(np.log1p).hist(figsize = (20,6))

plt.xlabel('Item Price')

plt.title('log1p Transformation')

sales.loc[:, 'item_price'] = sales.item_price.map(np.log1p)
unreasonable_data(sales.date_block_num)
count_price = sales.date_block_num.value_counts().sort_index(ascending = False)

plt.subplot(221)

count_price.hist(figsize = (20,5))

plt.xlabel('Date Block')

plt.title('Original Distribution')



count_price = sales.shop_id.value_counts().sort_index(ascending = False)

plt.subplot(222)

count_price.hist(figsize = (20,5))

plt.xlabel('shop_id')

plt.title('original distribution')



count_price = sales.item_id.value_counts().sort_index(ascending = False)

plt.subplot(223)

count_price.hist(figsize=(20,5))

plt.xlabel('item_id')

plt.title('original distribution')
l = list(item_cat.item_category_name)

l_cat = l



for ind in range(1,8):

    l_cat[ind] = 'Access'

    

for ind in range(10,18):

    l_cat[ind] = 'Consoles'



for ind in range(18,25):

    l_cat[ind] = 'Consoles Games'



for ind in range(26,28):

    l_cat[ind] = 'phone games'



for ind in range(28,32):

    l_cat[ind] = 'CD games'



for ind in range(32,37):

    l_cat[ind] = 'Card'



for ind in range(37,43):

    l_cat[ind] = 'Movie'



for ind in range(43,55):

    l_cat[ind] = 'Books'



for ind in range(55,61):

    l_cat[ind] = 'Music'



for ind in range(61,73):

    l_cat[ind] = 'Gifts'



for ind in range(73,79):

    l_cat[ind] = 'Soft'





item_cat['cats'] = l_cat

item_cat.head()

sales['date'] = pd.to_datetime(sales.date, format = '%d.%m.%Y')

sales.head()
p_df = sales.pivot_table(index = ['shop_id', 'item_id'], columns ='date_block_num', values ='item_cnt_day', aggfunc = 'sum').fillna(0.0)

p_df
# join with categories

sales_cleaned_df = p_df.reset_index()

sales_cleaned_df['shop_id'] = sales_cleaned_df.shop_id.astype('str')

item_to_cat_df = items.merge(item_cat[['item_category_id', 'cats']], how = 'inner', on = 'item_category_id')[['item_id', 'cats']]

#item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')

sales_cleaned_df = sales_cleaned_df.merge(item_to_cat_df, how = 'inner', on = 'item_id')



# Encode Categories

from sklearn import preprocessing



number = preprocessing.LabelEncoder()

sales_cleaned_df[['cats']] = number.fit_transform(sales_cleaned_df.cats)

sales_cleaned_df = sales_cleaned_df[['shop_id', 'item_id', 'cats'] + list(range(34))]

sales_cleaned_df.head()
import xgboost as xgb

param = {'max_depth':10,

        'subsample': 1,

        'min_child_weight': 0.5,

        'eta': 0.3,

        'num_round': 1000,

        'seed': 1, 

        'silend': 0,

        'eval_metric': 'rmse'}

progress = dict()

xgbtrain = xgb.DMatrix(sales_cleaned_df.iloc[:, (sales_cleaned_df.columns != 33)].values, sales_cleaned_df.iloc[:, sales_cleaned_df.columns == 33].values)

watchlist = [(xgbtrain, 'train-rmse')]



bst = xgb.train(param, xgbtrain)

preds = bst.predict(xgb.DMatrix(sales_cleaned_df.iloc[:,(sales_cleaned_df.columns != 33)].values))



from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(preds, sales_cleaned_df.iloc[:, sales_cleaned_df.columns == 33].values))

print(rmse)
xgb.plot_importance(bst)
test.info()
sales_cleaned_df.info()
apply_df = test

apply_df['shop_id'] = apply_df.shop_id.astype('str')

apply_df['item_id'] = apply_df.item_id.astype('str')

sales_cleaned_df['shop_id'] = sales_cleaned_df.shop_id.astype('str')

sales_cleaned_df['item_id'] = sales_cleaned_df.item_id.astype('str')

apply_df = test.merge(sales_cleaned_df, how = 'left', on = ['shop_id', 'item_id']).fillna(0.0)

apply_df.head()
# move to one month front

d = dict(zip(apply_df.columns[4:], list(np.array(list(apply_df.columns[4:])) -1)))



apply_df = apply_df.rename(d, axis = 1)

         
apply_df.head()
preds = bst.predict(xgb.DMatrix(apply_df.iloc[:, (apply_df.columns != 'ID') & (apply_df.columns != -1)].values))
# normalize prediction to [0-20]

preds = list(map(lambda x: min(20, max(x,0)), list(preds)))



sub_df = pd.DataFrame({'ID': apply_df.ID, 'item_cnt_month': preds})

sub_df.describe()
sub_df.to_csv('Submission_Predict Sales.csv', index = False)