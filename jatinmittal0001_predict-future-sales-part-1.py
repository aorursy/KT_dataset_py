import warnings

warnings.filterwarnings('ignore')

import gc

import pickle

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

from itertools import product

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler

from xgboost import XGBRegressor

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import plot_importance

from matplotlib.pyplot import figure

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Defining function which helps to reduce size of dataframes, so that kernel does not die

def reduce_size(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df
#Loading given dataframes

shops = pd.read_csv("../input/shops.csv")

item_categories = pd.read_csv("../input/item_categories.csv")

sales_train = pd.read_csv("../input/sales_train.csv")

items = pd.read_csv("../input/items.csv")

test = pd.read_csv("../input/test.csv")
# Check duplicated rows in train set

df = sales_train[sales_train.duplicated()]  # checks duplicate rows considering all columns

df
# to check how many times 1st row of above df is repeating

df.head(1)
#deleting df as it is not required elsewhere, it helps to reduce size of kernel

del df

gc.collect();
sales_train[(sales_train.date == "05.01.2013") & (sales_train.item_id==20130) & (sales_train.shop_id==54)]
#Dropping duplicates and keeping first occurence only

sales_train.drop_duplicates(keep = 'first', inplace = True) # keep: keeps first occurence as original, removes rest
items.head()


item_prices = sales_train.groupby('item_id').agg({'item_price':'mean'}).reset_index()

item_prices = pd.merge(item_prices, items, on = ['item_id'],how = 'left').drop('item_name',axis=1)

print(item_prices.head())
a = sales_train.shop_id.unique()

b = test.shop_id.unique()

l=0

for i in b:

    if i not in a:

        l = l +1

print(l)

print("total distinct shops in train set: ",len(a))

print("total distinct shops in test set: ",len(b))

print("Number of shops present in test set but not in train set: ", l)
a = sales_train.item_id.unique()

b = test.item_id.unique()

l=0

for i in b:

    if i not in a:

        l = l +1

print("total distinct items in train set: ",len(a))

print("total distinct items in test set: ",len(b))

print("Number of distinct items present in test set but not in train set: ", l)



del a 

del b

gc.collect();
item_categories.head()
sales_train.head()
item_prices_train_data = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_price':'mean'}).reset_index()

item_prices_train_data.head()
test.head()
print(shops.columns)

print(item_categories.columns)

print(sales_train.columns)

print(items.columns)

print(test.columns)
# insights

print('max. number of items sold {} and min are {}: '.format(sales_train['item_cnt_day'].max(),sales_train['item_cnt_day'].min()))

print('start date is {} and end date is {}'.format(sales_train['date'].min(), sales_train['date'].max()))

#note max date is actually October, 2015, python is not checking till last row
#checking the outliers in sales_train

plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=sales_train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(sales_train.item_price.min(), sales_train.item_price.max()*1.1)

sns.boxplot(x=sales_train.item_price)
#removing outliers

sales_train = sales_train[sales_train.item_price<50000]

sales_train = sales_train[sales_train.item_cnt_day<1001]
#number of rows having negative price of an item

print((sales_train.item_price<0).sum())

# fill it with median

#first find to which shop, item_id, month it belongs

sales_train[sales_train.item_price<0]
# filing negative price of item by its median value

median = sales_train[(sales_train.shop_id == 32) & (sales_train.item_id==2973) & (sales_train.date_block_num==4) 

            & (sales_train.item_price>0)].item_price.median()

sales_train.loc[sales_train.item_price<0, 'item_price'] = median
# getting unique shop_id and unique item_id

shop_unique = sales_train['shop_id'].unique()

item_id_unique = sales_train['item_id'].unique()

print("Number of unique  shops are {} and number of unique items are {}".format(len(shop_unique),

      len(item_id_unique)))

shops_len = shops.groupby('shop_name')['shop_id'].nunique()

#print(shops_len);
grouped = pd.DataFrame(sales_train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))

num_graph = 10

id_per_graph = math.ceil(grouped.shop_id.max() / num_graph)

count = 0

for i in range(5):

    for j in range(2):

        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])

        count += 1
print('Number of unique items in test data:', test.item_id.nunique())

print('Number of items present in test set but not in train set:',len(list(set(test.item_id) - set(test.item_id).intersection(set(sales_train.item_id)))))
item_details = pd.merge(item_categories, items[['item_id', 'item_category_id']], on ='item_category_id',how='left')

item_details.head()
item_grp = item_details['item_category_name'].apply(lambda x: str(x).split(' ')[0])

subtype = item_details['item_category_name'].str.split('-')

subtype = subtype.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_grp = pd.Categorical(item_grp).codes     # converting to label encoded feature

subtype = pd.Categorical(subtype).codes

item_details['item_group'] = item_grp

item_details['subtype'] = subtype

item_details  = item_details.drop(['item_category_name'],axis=1)
#filling item details dataframe with item prices. First by adding items from training set

# then filing NaN prices (obtained from test set) with mean

abc = pd.merge(item_details,item_prices, on = ['item_id','item_category_id'], how='left')

item_price_avg_subtype = abc.groupby(['item_group','subtype']).agg({'item_price':'mean'}).reset_index()

na = abc[abc.item_price.isnull()]

abc = abc[~abc.item_price.isnull()]

na.drop('item_price',axis=1, inplace=True)

na = pd.merge(na,item_price_avg_subtype, on=['item_group','subtype'], how = 'left')   #going granular to obtain better estimate of item prices

item_details = pd.concat([abc, na],axis=0)

item_details.head()
test_shops = test.shop_id.unique()

test_items = test.item_id.unique()

train_shops = sales_train.shop_id.unique()

train_items = sales_train.item_id.unique()
#rolling train data at month level

sales_monthly = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

sales_monthly.columns = ['item_cnt_month']

sales_monthly.reset_index(inplace=True)



#sales_monthly = pd.merge(sales_train, group, on=['date_block_num','shop_id','item_id'], how='left')

sales_monthly['item_cnt_month'] = (sales_monthly['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float16))
matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = sales_monthly[sales_monthly.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)
matrix.shape
sales_monthly = pd.merge(matrix,sales_monthly, on = ['date_block_num','shop_id','item_id'], how = 'left')

sales_monthly.shape
del matrix

del abc

del na

gc.collect();
# making test data similar to train data

test = test.drop('ID',axis=1)

test['date_block_num']=34
# joining train and test data to make feature engineering easy

cols = ['date_block_num','shop_id','item_id']

all_data = pd.concat([sales_monthly, test], ignore_index=True, sort=False, keys=cols)

all_data.fillna(0, inplace=True) # for 34th month and missing values which occured during combination of shop X item
#adding price data at date_block level from original sales data

all_data = pd.merge(all_data,item_prices_train_data, on = ['date_block_num','shop_id','item_id'], how = 'left')

#for items for which we don't have price data, filing it with price value created using means and then appending

all_data_non_null = all_data[~all_data.item_price.isnull()]

abc = all_data[all_data.item_price.isnull()]

abc.drop('item_price',axis=1,inplace=True)

abc = pd.merge(abc, item_details[['item_id','item_price']], on =['item_id'],how='left')



del all_data

gc.collect();



all_data = pd.concat([all_data_non_null,abc]).sort_values(by=['date_block_num','shop_id','item_id'])

item_details.drop('item_price',axis=1, inplace=True)

all_data = pd.merge(all_data, item_details, on =['item_id'],how='left')

#Deleting intermediate data to reduce kernel size

del all_data_non_null

del abc

del item_grp

del subtype

del item_prices_train_data

del sales_monthly

gc.collect();
# Pre-processing with Shop_name

all_data = pd.merge(all_data, shops[['shop_name','shop_id']], on ='shop_id',

                 how='left')

#if we see names of shop, they are all in Russian language. But there are few shops having same names but 

#extra some symbol has been added to them. we will make that same.



# Якутск Орджоникидзе, 56

all_data.loc[all_data.shop_id == 0, 'shop_id'] = 57

all_data.loc[all_data.shop_id == 0, 'shop_name'] = 'Якутск Орджоникидзе, 56'

test.loc[test.shop_id == 0, 'shop_id']=57

# Якутск ТЦ "Центральный"

all_data.loc[all_data.shop_id == 1, 'shop_id'] = 58

all_data.loc[all_data.shop_id == 1, 'shop_name'] = 'Якутск ТЦ "Центральный"'

test.loc[test.shop_id == 1, 'shop_id']=58

# Жуковский ул. Чкалова 39м²

all_data.loc[all_data.shop_id == 10, 'shop_id'] = 11

all_data.loc[all_data.shop_id == 10, 'shop_name'] = 'Жуковский ул. Чкалова 39м²'

test.loc[test.shop_id == 10, 'shop_id']= 11



city = all_data['shop_name'].apply(lambda x: str(x).split(' ')[0])

city = pd.Categorical(city).codes

all_data['city'] = city

all_data  = all_data.drop(['shop_name'],axis=1)

print(all_data.shape)

all_data.tail()
#Reducing size of main dataframe

total_before_opti = sum(all_data.memory_usage())

reduce_size(all_data);



print("Memory usage before optimization :", str(round(total_before_opti/1000000000,2))+'GB')

print("Memory usage after optimization :", str(round(sum(all_data.memory_usage())/1000000000,2))+'GB')

print("We reduced the dataframe size by",str(round(((total_before_opti - sum(all_data.memory_usage())) /total_before_opti)*100,2))+'%')
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id', col]]



    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id',col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        #df.loc[i*, col+'_lag_'+str(i)] = shifted[col+'_lag_'+str(i)]

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

        del shifted

    del tmp

    gc.collect();

    return df
group1 = all_data.groupby(['date_block_num']).agg({'item_cnt_month':['mean']})

group1.columns = ['db_avg_items_sold']

group1.reset_index(inplace=True)



group2 = all_data.groupby(['date_block_num','shop_id']).agg({'item_cnt_month':['mean']})

group2.columns = ['db_shop_avg_items_sold']

group2.reset_index(inplace=True)



group3 = all_data.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':['mean']})

group3.columns = ['db_cat_avg_items_sold']

group3.reset_index(inplace=True)



group4 = all_data.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month':['mean']})

group4.columns = ['db_shop_cat_avg_items_sold']

group4.reset_index(inplace=True)



group5 = all_data.groupby(['date_block_num','item_id']).agg({'item_cnt_month':['mean']})

group5.columns = ['db_item_id_items_sold']

group5.reset_index(inplace=True)



group6 = all_data.groupby(['date_block_num','shop_id','city']).agg({'item_cnt_month':['mean']})

group6.columns = ['db_shop_city_avg_items_sold']

group6.reset_index(inplace=True)



group7 = all_data.groupby(['date_block_num','city']).agg({'item_cnt_month':['mean']})

group7.columns = ['db_city_avg_items_sold']

group7.reset_index(inplace=True)



group8 = all_data.groupby(['date_block_num','item_category_id','city']).agg({'item_cnt_month':['mean']})

group8.columns = ['db_cat_city_avg_items_sold']

group8.reset_index(inplace=True)
#Merging lag_features with main data



# group-1

all_data = pd.merge(all_data, group1, on = ['date_block_num'], how = 'left')

#all_data['db_avg_items_sold'] = all_data['db_avg_items_sold']

all_data = lag_feature(all_data, [1], 'db_avg_items_sold')

all_data.drop('db_avg_items_sold',axis=1,inplace=True)

del group1

gc.collect();

#reduce_size(all_data)



# group-2

all_data = pd.merge(all_data, group2, on = ['date_block_num','shop_id'], how = 'left')

#all_data['db_shop_avg_items_sold'] = all_data['db_shop_avg_items_sold']

all_data = lag_feature(all_data, [1], 'db_shop_avg_items_sold')                                 # initially took 3,6,12 lag but they were not important features, checked after running model

all_data.drop('db_shop_avg_items_sold',axis=1,inplace=True)

del group2

gc.collect();

#reduce_size(all_data)



# group-3

all_data = pd.merge(all_data, group3, on = ['date_block_num','item_category_id'], how = 'left')

#all_data['db_cat_avg_items_sold'] = all_data['db_cat_avg_items_sold']

all_data = lag_feature(all_data, [1,2], 'db_cat_avg_items_sold')

all_data.drop('db_cat_avg_items_sold',axis=1,inplace=True)

del group3

gc.collect();

reduce_size(all_data);



# group-4

all_data = pd.merge(all_data, group4, on = ['date_block_num','shop_id','item_category_id'], how = 'left')

#all_data['db_shop_cat_avg_items_sold'] = all_data['db_shop_cat_avg_items_sold']

all_data = lag_feature(all_data, [1,2,3], 'db_shop_cat_avg_items_sold')

all_data.drop('db_shop_cat_avg_items_sold',axis=1,inplace=True)

del group4

gc.collect();

#reduce_size(all_data)



# group-5

all_data = pd.merge(all_data, group5, on = ['date_block_num','item_id'], how = 'left')

#all_data['db_cat_subtype_avg_items_sold'] = all_data['db_cat_subtype_avg_items_sold']

all_data = lag_feature(all_data, [1,2,3,6,12], 'db_item_id_items_sold')

all_data.drop('db_item_id_items_sold',axis=1,inplace=True)



del group5

gc.collect();



# group-6

all_data = pd.merge(all_data, group6, on = ['date_block_num','shop_id','city'], how = 'left')

#all_data['db_shop_city_avg_items_sold'] = all_data['db_shop_city_avg_items_sold']

all_data = lag_feature(all_data, [1,2,3,6,12], 'db_shop_city_avg_items_sold')

all_data.drop('db_shop_city_avg_items_sold',axis=1,inplace=True)

del group6

gc.collect();



# group-7

all_data = pd.merge(all_data, group7, on = ['date_block_num','city'], how = 'left')

#all_data['db_city_avg_items_sold'] = all_data['db_city_avg_items_sold']

all_data = lag_feature(all_data, [1], 'db_city_avg_items_sold')                                  # initially took 3,6,12 lag but they were not important features, checked after running model

all_data.drop('db_city_avg_items_sold',axis=1,inplace=True)

del group7

gc.collect();



# group-8

all_data = pd.merge(all_data, group8, on = ['date_block_num','item_category_id','city'], how = 'left')

#all_data['db_city_avg_items_sold'] = all_data['db_city_avg_items_sold']

all_data = lag_feature(all_data, [1], 'db_cat_city_avg_items_sold')                            # initially took 2,3,6,12 lag but they were not important features, checked after running model

all_data.drop('db_cat_city_avg_items_sold',axis=1,inplace=True)

reduce_size(all_data);



del group8

gc.collect();
# Creating more features

item_max_cnt = all_data.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_month':np.max})

item_min_cnt = all_data.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_month':np.min})

item_max_cnt.columns = ['max_cnt']

item_min_cnt.columns = ['min_cnt']

item_max_cnt.reset_index(inplace=True)

item_min_cnt.reset_index(inplace=True)





all_data = pd.merge(all_data, item_max_cnt, on = ['date_block_num','shop_id','item_id'], how = 'left')

all_data = pd.merge(all_data, item_min_cnt, on = ['date_block_num','shop_id','item_id'], how = 'left')



del item_max_cnt

del item_min_cnt

gc.collect();



all_data = lag_feature(all_data, [1,3,6], 'max_cnt') 

all_data = lag_feature(all_data, [1,3,6], 'min_cnt') 



all_data.drop(['max_cnt','min_cnt'], axis=1, inplace=True)
#Adding month variable, it will also help in mapping holidays, number of days etc

all_data['month'] = (all_data['date_block_num'] % 12)+1    #date_block_num starts from 1
training_data_prices = all_data[(all_data.date_block_num<34)]

# we don't  need for date_block ==34, because we are making only lag features so for last month, 

# lag can be created using previous months data



price_group = training_data_prices.groupby(['date_block_num', 'item_id']).agg({'item_price':'mean'})

price_group.columns = ['db_item_avg_price']



#del training_data_prices

#gc.collect();



all_data = pd.merge(all_data,price_group, on = ['date_block_num', 'item_id'], how='left')





del price_group

gc.collect();

# will fill price data for test set too



all_data = lag_feature(all_data, [1,3,6,12], 'db_item_avg_price')

lags = [1,3,6,12]

for i in lags:

    all_data['delta_price_lag_'+str(i)] = (all_data['db_item_avg_price'] - all_data['db_item_avg_price_lag_' + str(i)])/(all_data['db_item_avg_price'])



    

#making revenue features

all_data['revenue'] = all_data['item_cnt_month']*all_data['item_price']

revenue_shop = all_data.groupby(['date_block_num','shop_id','item_id']).agg({'revenue':'sum'})

revenue_shop.columns = ['revenue_shop']

all_data = pd.merge(all_data,revenue_shop, on = ['date_block_num','shop_id','item_id'], how='left')  # will fill price data for test set too





all_data = lag_feature(all_data, [1,2], 'revenue_shop')

all_data = lag_feature(all_data, [1], 'revenue')  # initialy took multiple lags, but after seeing feature importance

                                                    #taking only lag1



all_data.drop(['db_item_avg_price','revenue','revenue_shop'],axis=1,inplace=True)
all_data['item_shop_months_since_first_sale'] = all_data['date_block_num'] - all_data.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

all_data['item_months_since_first_sale'] = all_data['date_block_num'] - all_data.groupby('item_id')['date_block_num'].transform('min')
reduce_size(all_data);
cache = {}

all_data['item_shop_last_sale'] = 0

all_data['item_shop_last_sale'] = all_data['item_shop_last_sale'].astype(np.int8)

for idx, row in all_data.iterrows():    

    key = str(row.item_id)+' '+str(row.shop_id)

    if key not in cache:

        if row.item_cnt_month!=0:

            cache[key] = row.date_block_num

    else:

        last_date_block_num = cache[key]

        all_data.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num

        cache[key] = row.date_block_num



del cache

gc.collect();
# was killing the kernel because of size

'''

cache = {}

all_data['item_last_sale'] = 0

all_data['item_last_sale'] = all_data['item_last_sale'].astype(np.int8)

for idx, row in all_data.iterrows():    

    key = row.item_id

    if key not in cache:

        if row.item_cnt_month!=0:

            cache[key] = row.date_block_num

    else:

        last_date_block_num = cache[key]

        if row.date_block_num>last_date_block_num:

            all_data.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num

            cache[key] = row.date_block_num 

            

del cache

gc.collect();

'''
lags = [1,3,6,12]

def select_first_non_null(row):

    for i in lags:

        if row['delta_price_lag_'+str(i)]:

            return row['delta_price_lag_'+str(i)]

    return 0

    

all_data['delta_price_lag'] = all_data.apply(select_first_non_null, axis=1).astype(np.float16)

all_data['delta_price_lag'].fillna(0, inplace=True)



all_data.item_cnt_month = all_data.item_cnt_month.astype('float32')
#Visualize item sales for each calendar month

a = all_data.groupby('date_block_num').agg({'item_cnt_month':'sum'})

plt.plot(a)



#del a 

#gc.collect();
holidays_dict1 = {1 : 7, 2:3, 3:2, 4:2, 5:7, 6:3,7:0, 8:1, 9:3, 10:0, 11:2, 12:6}  #source: mentioned above

holidays_dict2 = {1 : 6, 2:3, 3:2, 4:8, 5:3, 6:3,7:2, 8:8, 9:4, 10:8, 11:5, 12:4}  #source: from other kernels

num_days_dict = {1:31 ,2:28 ,3:31,4:30 ,5:31,6: 30,7:31,8:31,9:30,10:31,11:30,12:31}

is_dec = {1:0 ,2:0 ,3:0,4:0 ,5:0,6: 0,7:0,8:0,9:0,10:0,11:0,12:1}

'''

# the following feature captures inflation rate starting from Jan 2013, Source: Internet

inflation_cpi = {1: 7.07, 2: 7.28, 3: 7.02, 4: 7.23, 5: 7.38, 6: 6.88, 7: 6.45, 8: 6.49, 9: 6.13, 10: 6.25, 11: 6.5, 12: 6.45, 13: 6.05, 14: 6.19, 15: 6.91, 16: 7.33, 17: 7.59

                 ,18: 7.8, 19: 7.45, 20: 7.55, 21: 8.03, 22: 8.29, 23: 9.06, 24: 11.36, 25: 14.97, 26: 16.71, 27: 16.93, 28: 16.42, 29: 15.78, 30: 15.29, 31: 15.64, 32: 15.77, 33: 15.68

                 , 34: 15.59, 35: 14.98, 36: 12.91}

'''

all_data['num_holidays'] = all_data['month'].map(holidays_dict2) #giving better performance

all_data['num_days'] = all_data['month'].map(num_days_dict)

#all_data['inflation_cpi'] = all_data['date_block_num'].map(inflation_cpi)

all_data['is_dec'] = all_data['month'].map(is_dec)

all_data.head()
#deleting files which are not required

del shops

del items

del item_categories

del city

gc.collect();
reduce_size(all_data);
all_data1 = all_data[all_data.date_block_num>=12]



del all_data

gc.collect();
def fillna(df):

    for col in df.columns:

        if (('_lag_' in col) & (df[col].isnull().any())):

            if ('_items_sold' in col):

                df[col].fillna(0,inplace=True)

            if ('_price_lag_' in col):

                df[col].fillna(0,inplace=True)

            if ('revenue_shop_lag' in col):

                df[col].fillna(0,inplace=True)

            if ('revenue_lag' in col):

                df[col].fillna(0,inplace=True)

    return df
all_data1 = fillna(all_data1)

all_data1.head(2)
# Encoding-1



n_row_test = all_data1[all_data1.date_block_num==34].shape[0]

n_row_train = all_data1[all_data1.date_block_num<34].shape[0]



mean_item_cnt = all_data1.item_cnt_month.mean()



#Calculate a mapping: {item_id: target_mean}

item_id_target_mean = all_data1[:n_row_train].groupby('item_id').item_cnt_month.mean()



#In our non-regularized case we just *map* the computed means to the `item_id`'s

all_data1['item_target_enc'] = all_data1['item_id'].map(item_id_target_mean)



#Fill NaNs

all_data1['item_target_enc'].fillna(mean_item_cnt, inplace=True) 





#Calculate a mapping: {city: target_mean}

item_id_target_mean = all_data1[:n_row_train].groupby('city').item_cnt_month.mean()



#In our non-regularized case we just *map* the computed means to the `item_id`'s

all_data1['city_target_enc'] = all_data1['city'].map(item_id_target_mean)



#Fill NaNs

all_data1['city_target_enc'].fillna(mean_item_cnt, inplace=True) 





#Calculate a mapping: {month: target_mean}

item_id_target_mean = all_data1[:n_row_train].groupby('month').item_cnt_month.mean()



#In our non-regularized case we just *map* the computed means to the `item_id`'s

all_data1['month_target_enc'] = all_data1['month'].map(item_id_target_mean)



#Fill NaNs

all_data1['month_target_enc'].fillna(mean_item_cnt, inplace=True) 



#Calculate a mapping: {item_category_id: target_mean}

item_id_target_mean = all_data1[:n_row_train].groupby('item_category_id').item_cnt_month.mean()



#In our non-regularized case we just *map* the computed means to the `item_id`'s

all_data1['item_id_target_enc'] = all_data1['item_category_id'].map(item_id_target_mean)



#Fill NaNs

all_data1['item_id_target_enc'].fillna(mean_item_cnt, inplace=True) 

#Encoding-2



cumsum = all_data1.groupby('city').item_cnt_month.cumsum() - all_data1['item_cnt_month']

cumcnt = all_data1.groupby('city').cumcount()

all_data1['city_target_enc_2'] = cumsum/cumcnt



# Fill NaNs

all_data1['city_target_enc_2'].fillna(mean_item_cnt, inplace=True) 



cumsum = all_data1.groupby('item_id').item_cnt_month.cumsum() - all_data1['item_cnt_month']

cumcnt = all_data1.groupby('item_id').cumcount()

all_data1['item_id_target_enc_2'] = cumsum/cumcnt



# Fill NaNs

all_data1['item_id_target_enc_2'].fillna(mean_item_cnt, inplace=True) 



cumsum = all_data1.groupby('month').item_cnt_month.cumsum() - all_data1['item_cnt_month']

cumcnt = all_data1.groupby('month').cumcount()

all_data1['month_target_enc_2'] = cumsum/cumcnt



# Fill NaNs

all_data1['month_target_enc_2'].fillna(mean_item_cnt, inplace=True) 



cumsum = all_data1.groupby('subtype').item_cnt_month.cumsum() - all_data1['item_cnt_month']

cumcnt = all_data1.groupby('subtype').cumcount()

all_data1['subtype_target_enc_2'] = cumsum/cumcnt



# Fill NaNs

all_data1['subtype_target_enc_2'].fillna(mean_item_cnt, inplace=True) 



cumsum = all_data1.groupby('item_category_id').item_cnt_month.cumsum() - all_data1['item_cnt_month']

cumcnt = all_data1.groupby('item_category_id').cumcount()

all_data1['item_cat_target_enc_2'] = cumsum/cumcnt



# Fill NaNs

all_data1['item_cat_target_enc_2'].fillna(mean_item_cnt, inplace=True) 



cumsum = all_data1.groupby('item_group').item_cnt_month.cumsum() - all_data1['item_cnt_month']

cumcnt = all_data1.groupby('item_group').cumcount()

all_data1['item_group_target_enc_2'] = cumsum/cumcnt



# Fill NaNs

all_data1['item_group_target_enc_2'].fillna(mean_item_cnt, inplace=True)



del cumsum

del cumcnt

gc.collect();
reduce_size(all_data1);
all_data1.head(2)
# saving data

model_file = "all_data1"

with open(model_file,mode='wb') as model_f:

    pickle.dump(all_data1,model_f)