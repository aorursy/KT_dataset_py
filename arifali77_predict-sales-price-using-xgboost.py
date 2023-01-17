# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

plt.style.use('ggplot') 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
items  = pd.read_csv('../input/items.csv')

train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

item_category = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')
def eda(data):

    print("----------Top-5- Record----------")

    print(data.head(5))

    print("-----------Information-----------")

    print(data.info())

    print("-----------Data Types-----------")

    print(data.dtypes)

    print("----------Missing value-----------")

    print(data.isnull().sum())

    print("----------Null value-----------")

    print(data.isna().sum())

    print("----------Shape of Data----------")

    print(data.shape)



def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

    

def drop_duplicate(data, subset):

    print('Before drop shape:', data.shape)

    before = data.shape[0]

    data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check

    data.reset_index(drop=True, inplace=True)

    print('After drop shape:', data.shape)

    after = data.shape[0]

    print('Total Duplicate:', before-after)
# sales train insights

eda(train)

graph_insight(train)
# Drop Duplicate Data

subset = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']

drop_duplicate(train, subset = subset)
# test insight

eda(test)

graph_insight(test)
eda(items)

graph_insight(items)
eda(item_category)

# graph_insight(item_category)
eda(shops)

# graph_insight(shops)
def unresanable_data(data):

    print("Min Value:",data.min())

    print("Max Value:",data.max())

    print("Average Value:",data.mean())

    print("Center Point of Data:",data.median())
# -1 and 307980 looks like outliers, let's delete them

print('before train shape:', train.shape)

train = train[(train.item_price > 0) & (train.item_price < 300000)]

print('after train shape:', train.shape)
train.groupby('date_block_num').sum()['item_cnt_day'].hist(figsize = (20,4))

plt.title('Sales per month histogram')

plt.xlabel('Price')



plt.figure(figsize = (20,4))

sns.tsplot(train.groupby('date_block_num').sum()['item_cnt_day'])

plt.title('Sales per month')

plt.xlabel('Price')
unresanable_data(train['item_price'])

count_price = train.item_price.value_counts().sort_index(ascending=False)

plt.subplot(221)

count_price.hist(figsize=(20,6))

plt.xlabel('Item Price', fontsize=20);

plt.title('Original Distiribution')



plt.subplot(222)

train.item_price.map(np.log1p).hist(figsize=(20,6))

plt.xlabel('Item Price');

plt.title('log1p Transformation')

train.loc[:,'item_price'] = train.item_price.map(np.log1p)
#unresanable_data(train['date_block_num'])

count_price = train.date_block_num.value_counts().sort_index(ascending=False)

plt.subplot(221)

count_price.hist(figsize=(20,5))

plt.xlabel('Date Block');

plt.title('Original Distiribution')



count_price = train.shop_id.value_counts().sort_index(ascending=False)

plt.subplot(222)

count_price.hist(figsize=(20,5))

plt.xlabel('shop_id');

plt.title('Original Distiribution')



count_price = train.item_id.value_counts().sort_index(ascending=False)

plt.subplot(223)

count_price.hist(figsize=(20,5))

plt.xlabel('item_id');

plt.title('Original Distiribution')
item_category.head()
l = list(item_category.item_category_name)

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





item_category['cats'] = l_cat

item_category.head()
train['date'] = pd.to_datetime(train.date,format="%d.%m.%Y")

train.head()
## Pivot by monht to wide format

p_df = train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

p_df.head()
## Join with categories

train_cleaned_df = p_df.reset_index()

train_cleaned_df['shop_id']= train_cleaned_df.shop_id.astype('str')

train_cleaned_df['item_id']= train_cleaned_df.item_id.astype('str')



item_to_cat_df = items.merge(item_category[['item_category_id','cats']], how="inner", on="item_category_id")[['item_id','cats']]

item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')



train_cleaned_df = train_cleaned_df.merge(item_to_cat_df, how="inner", on="item_id")





# Encode Categories

from sklearn import preprocessing



number = preprocessing.LabelEncoder()

train_cleaned_df[['cats']] = number.fit_transform(train_cleaned_df.cats)

train_cleaned_df = train_cleaned_df[['shop_id', 'item_id', 'cats'] + list(range(34))]

train_cleaned_df.head()
import xgboost as xgb

param = {'max_depth':10, 

         'subsample':1,

         'min_child_weight':0.5,

         'eta':0.3, 

         'num_round':1000, 

         'seed':1,

         'silent':0,

         'eval_metric':'rmse'}



progress = dict()

xgbtrain = xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values, train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values)

watchlist  = [(xgbtrain,'train-rmse')]



bst = xgb.train(param, xgbtrain)

preds = bst.predict(xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values))

from sklearn.metrics import mean_squared_error 

rmse = np.sqrt(mean_squared_error(preds,train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values))

print(rmse)
xgb.plot_importance(bst)
apply_df = test

apply_df['shop_id']= apply_df.shop_id.astype('str')

apply_df['item_id']= apply_df.item_id.astype('str')



apply_df = test.merge(train_cleaned_df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

apply_df.head()
# Move to one month front

d = dict(zip(apply_df.columns[4:],list(np.array(list(apply_df.columns[4:])) - 1)))



apply_df  = apply_df.rename(d, axis = 1)
preds = bst.predict(xgb.DMatrix(apply_df.iloc[:, (apply_df.columns != 'ID') & (apply_df.columns != -1)].values))
# Normalize prediction to [0-20]

preds = list(map(lambda x: min(20,max(x,0)), list(preds)))

sub_df = pd.DataFrame({'ID':apply_df.ID,'item_cnt_month': preds })

sub_df.describe()
sub_df.head(20)
sub_df.to_csv('Submission_Predict Sales.csv',index=False)