# This Python 3 environment with many helpful analytics libraries installed

# defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#Kaggle 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



# A list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Results are saved as output.
#open and convert data to dataframe for visualization and exploration 

import pandas as pd

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
items.head()
items.describe()
items.shape
items.dtypes
items.info()

#this tells me if there are null entries in the data. There is none here. The shape and the entries match.
item_categories.head()
sales_train.head() 
#correlations among columns in that dataset

sales_train.corr()
#plot data for visualization

import matplotlib.pyplot as plt

%matplotlib inline

sales_train.corr().plot.bar()
sample_submission.head() 
shops.head() 
test.head()
def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

    
graph_insight(sales_train)
def drop_duplicate(data, subset):

    print('Before drop shape:', sales_train.shape)

    before = sales_train.shape[0]

    sales_train.drop_duplicates(subset,keep='first', inplace=True) #a list for all columns for duplicate check

    sales_train.reset_index(drop=True, inplace=True)

    print('After drop shape:', sales_train.shape)

    after = sales_train.shape[0]

    print('Total Duplicate:', before-after)
# Drop Duplicate Data

subset = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']

drop_duplicate(sales_train, subset = subset)
#remove outliers

print('before train shape:', sales_train.shape)

train = sales_train[(sales_train.item_price > 0) & (sales_train.item_price < 300000)] #removes a negative entry and a far our entry

print('after train shape:', train.shape)
#convert date column

train['date'] = pd.to_datetime(train.date,format="%d.%m.%Y")

train.head(3)
#Pivot tables - create new tables from existing tables

# Pivot by month in wide format

df1 = train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

df1.head()
list1 = list(item_categories.item_category_name)

listcat = list1



for ind in range(1,8):

    listcat[ind] = 'Access'



for ind in range(10,18):

    listcat[ind] = 'Consoles'



for ind in range(18,25):

    listcat[ind] = 'Consoles Games'



for ind in range(26,28):

    listcat[ind] = 'phone games'



for ind in range(28,32):

    listcat[ind] = 'CD games'



for ind in range(32,37):

    listcat[ind] = 'Card'



for ind in range(37,43):

    listcat[ind] = 'Movie'



for ind in range(43,55):

    listcat[ind] = 'Books'



for ind in range(55,61):

    listcat[ind] = 'Music'



for ind in range(61,73):

    listcat[ind] = 'Gifts'



for ind in range(73,79):

    listcat[ind] = 'Soft'





item_categories['cats'] = listcat

item_categories.head()
## merge with categories (cats)

train_cleaned_df = df1.reset_index()

train_cleaned_df['shop_id']= train_cleaned_df.shop_id.astype('str')

train_cleaned_df['item_id']= train_cleaned_df.item_id.astype('str')



item_to_cat_df = items.merge(item_categories[['item_category_id','cats']], how="inner", on="item_category_id")[['item_id','cats']]

item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')



train_cleaned_df = train_cleaned_df.merge(item_to_cat_df, how="inner", on="item_id")
# label-encode categorical data

from sklearn import preprocessing



labl = preprocessing.LabelEncoder()

train_cleaned_df[['cats']] = labl.fit_transform(train_cleaned_df.cats)

train_cleaned_df = train_cleaned_df[['shop_id', 'item_id', 'cats'] + list(range(34))]

train_cleaned_df.head()
import xgboost as xgb

param = {'max_depth':10, #max_depth: determines how deeply each tree is allowed to grow during any boosting round.

         'subsample':1, #subsample: percentage of samples used per tree. Low value can lead to underfitting.

         'min_child_weight':0.5,

         'eta':0.3, 

         'num_round':1000, #number of trees built

         'seed':1, #for reproducibility of results.

         'silent':0,

         'eval_metric':'rmse'} #the evaluation metrics. Here is Root Mean Square Error

progress = dict()

xgbtrain = xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values, train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values)

watchlist  = [(xgbtrain,'train-rmse')]



bst = xgb.train(param, xgbtrain)

preds = bst.predict(xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values))

from sklearn.metrics import mean_squared_error 

rmse = np.sqrt(mean_squared_error(preds,train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values))

print(rmse)
df2 = test

df2['shop_id']= df2.shop_id.astype('str')

df2['item_id']= df2.item_id.astype('str')



df2= test.merge(train_cleaned_df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

df2.head(3)
d = dict(zip(df2.columns[4:],list(np.array(list(df2.columns[4:])) - 1)))

df2  = df2.rename(d, axis = 1)
#prediction

predic = bst.predict(xgb.DMatrix(df2.iloc[:, (df2.columns != 'ID') & (df2.columns != -1)].values))
### Normalize prediction into [0,20] range

predic = list(map(lambda x: min(20,max(x,0)), list(predic)))

df3 = pd.DataFrame({'ID':df2.ID,'item_cnt_month': predic })

df3.describe()
df3
#to csv

df3.to_csv(r'C:\Users\myrna\Desktop\csv2.csv', index = None, header=True)