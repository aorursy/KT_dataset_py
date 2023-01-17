# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

item_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
print('ITEMS')

display(items.head())



print('ITEM CATEGORIES')

display(item_cat.head())



print('SHOPS')

display(shops.head())



print('SALES TRAIN')

display(sales_train.head())



print('TEST')

display(test.head())



print('SAMPLE SUBMISSION')

display(sample_submission.head())
# Exploring Sample train dataset

sales_train.info()
# Checking missing values for sales train and test dataset

print('Sales train')

display(sales_train.isnull().sum())



print('test')

display(test.isnull().sum())
print('Sales train')

display(sales_train.describe())



print('Test')

display(test.describe())
# Removing duplicates from Sales train dataset

subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']

sales_train.duplicated(subset=subset).value_counts()
sales_train.drop_duplicates(subset=subset,inplace=True)
#Checking for negative values in item_price and item_cnt_day and dropping them as it contains only 1 negative value.

print('Negative values in item_price')

display(sales_train[sales_train['item_price']<0])



print('Negative values in item_cnt_day')

display(sales_train[sales_train['item_cnt_day']<0])

sales_train = sales_train[sales_train['item_price']>0]

sales_train = sales_train[sales_train['item_cnt_day']>0]

# Finding Outliers for item price and item_cnt_day

sns.boxplot(sales_train['item_price'])
sns.boxplot(sales_train['item_cnt_day'])
#Dropping Ouliers

def drop_out(df,feature,high_percentile = .99):

    df_shape = df.shape[0]         #sales train df size before dropping

    max_val = df[feature].quantile(high_percentile)      #Percentile value

    print('Dropping Outliers for ... {}'.format(feature))

    df = df[df[feature] < max_val]

    print(str(df_shape - df.shape[0]) + ' ' + feature + ' values over ' + str(max_val) + ' have been removed' )

    return df

    
#Dropping outliers for item_price

sales_train = drop_out(sales_train,'item_price')
#Dropping outliers for item_cnt_day

sales_train = drop_out(sales_train,'item_cnt_day')
# Creating new dataframe with item_price feature group by shop_id and item_id to get price for each item per shop.

# We can use this dataframe to create item_price feature for the test dataset.
shop_price_df = sales_train[['shop_id','item_id','item_price']]

shop_price_df = shop_price_df.groupby(['shop_id','item_id']).apply(lambda df: df['item_price'][-2:].mean())

shop_price_df = shop_price_df.to_frame(name='item_price')

shop_price_df
# Merge this dataframe with test dataframe to create item_price feature in test dataset.

test = pd.merge(test,shop_price_df,how='left', left_on=['shop_id','item_id'], right_on=['shop_id','item_id'])

test.head()
#check null values

test['item_price'].isnull().sum()      
#Split date feature to month and year from sales_train dataset

sales_train['month'] = [date.split('.')[1] for date in sales_train['date']]

sales_train['year'] = [date.split('.')[2] for date in sales_train['date']]



sales_train.drop(['date','date_block_num'],axis=1,inplace=True)



#create month and year features fot test dataset

test['month'] = 11

test['year'] = 2015
#change item_cnt_day to item_cnt_month 

sales_train_monthly = sales_train.groupby(['year','month','shop_id','item_id'],

                                          as_index=False)[['item_cnt_day']].sum()

sales_train_monthly.rename(columns={'item_cnt_day':'item_cnt_month'},inplace=True)



sales_train_monthly = pd.merge(sales_train_monthly,shop_price_df,how='left',

                               left_on=['shop_id','item_id'],right_on=['shop_id','item_id'])

sales_train_monthly.head()
sales_train = sales_train_monthly

sales_train.head()
test = test.reindex(columns=['ID','year','month','shop_id','item_id','item_price'])

test.head()
#Extracting main categories 

item_cat['main categories'] = [x.split('-')[0] for x in item_cat['item_category_name']]



# Some items dont have sub category for them we will use None as a sub category

sub_cat=[]

for i in range(len(item_cat)):

    try:

        sub_cat.append(item_cat['item_category_name'][i].split('-')[1])

    except IndexError as e:

        sub_cat.append('None')



item_cat['sub categories'] = sub_cat

item_cat.drop(['item_category_name'],axis=1,inplace=True)

item_cat.head()
items = pd.merge(items,item_cat,how='left')



#drop item_name and item_category_id

items.drop(['item_name','item_category_id'],axis=1,inplace=True)

items.head()
# Merge items to test and sales_train dataset

sales_train = pd.merge(sales_train,items,how='left')

test = pd.merge(test,items,how='left')
from string import punctuation



#replace all the punctuations from shop_name feature

shops['shop_name_cleaned'] = shops['shop_name'].apply(lambda s:''.join([x for x in s if x not in punctuation]))



#Extract shop city name

shops['shop_city'] = shops['shop_name_cleaned'].apply(lambda s:s.split()[0])



#Extract shop type

shops['shop_type'] = shops['shop_name_cleaned'].apply(lambda s:s.split()[1])



#Extract shop name

shops['shop_name'] = shops['shop_name_cleaned'].apply(lambda s:s.split()[2:])



#Dropping shop_name_cleaned

shops.drop(['shop_name_cleaned'],axis=1,inplace=True)



shops.head()
#Merging sales_train and test dataset with shops dataset

sales_train = pd.merge(sales_train,shops,how='left')

test = pd.merge(test,shops,how='left')
print('SALES TRAIN')

display(sales_train.head())



print('TEST')

display(test.head())
#fill missing values with median of each main_category and sub_category

test['item_price'] = test.groupby(['main categories',

                                   'sub categories'])['item_price'].apply(lambda df: df.fillna(df.median()))

test['item_price'].isnull().sum()
# Fill missing values with median of each sub category

test['item_price'] = test.groupby(['sub categories'])['item_price'].apply(lambda df : df.fillna(df.median()))

test['item_price'].isnull().sum()
test[test['item_price'].isnull()]
#fill missing values with median of main_category and sub_category from sales_train dataset



filler = sales_train[(sales_train['main categories'] == 'PC') & (sales_train['sub categories'] == 'Гарнитуры/Наушники')]['item_price'].median()

filler = 0

test['item_price'].fillna(filler, inplace=True)
test['item_price'].isnull().sum()
# Clipping the item_cnt_month value to [0,20] range

sales_train['item_cnt_month'] = sales_train['item_cnt_month'].clip(0,20)
# Defining a target array and dropping it from sales_train dataset

target_aaray = sales_train['item_cnt_month']

sales_train.drop(['item_cnt_month'],axis=1,inplace=True)



#Dropping ID column from test dataset

test_id = test['ID']

test.drop(['ID'],axis=1,inplace=True)

#Drop shop_id and item_id from sales_train and test dataset

sales_train.drop(['shop_id','item_id'],axis=1,inplace=True)

test.drop(['shop_id','item_id'],axis=1,inplace=True)
#Reducing memory usage

def downcast_dtypes(df):

    # Selecting columns to downcast

    flt_cols = [c for c in df if df[c].dtype == 'float64']

    int_cols = [d for d in df if df[d].dtype == 'int64']

    

    #Downcasting

    df[flt_cols] = df[flt_cols].astype(np.float16)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df
downcast_dtypes(sales_train)

downcast_dtypes(test)



print('SALES TRAIN')

display(sales_train.head())



print('TEST')

display(test.head())
sales_train.info()
#Check for null values in test and sales_train dataset

print('Missing data in sales train: ',sales_train.isnull().any().sum())

print('Missing data in test: ',test.isnull().any().sum())
#Normality test function

def norm_test(data,alpha=0.05):

    from scipy import stats

    statistic,p_val = stats.normaltest(data)

    

    #null hypothesis: array comes from a normal distribution

    if p_val < alpha:  

        #The null hypothesis can be rejected

        is_normal_dist = False

    else:

        #The null hypothesis cannot be rejected

        is_normal_dist = True

    

    return is_normal_dist
# Check normality of all numerical features and change them to normal distribution

for i in sales_train.columns:

    if sales_train[i].dtype != 'object':

        if norm_test(sales_train[i]) == False:

            sales_train[i] = np.log1p(sales_train[i])

            test[i] = np.log1p(test[i])
target_aaray = np.log1p(target_aaray)
#ENCODING

ord_enc = OrdinalEncoder()

arr = sales_train.to_numpy()

#x = sales_train.to_numpy(dtype=str)

print(arr)

X = ord_enc.fit_transform(arr)

Y = target_aaray



#X_predict = ord_enc.fit_transform(test.to_numpy())
x_train,x_test,y_train,y_test = train_test_split(sales_train,target_aaray,test_size=0.1,random_state=0)
#Create a model

xg_model = XGBRegressor()



#Fitting model

xg_model.fit(x_train, y_train, eval_metric="rmse", eval_set=[(x_train, y_train), (x_test, y_test)], verbose=True, early_stopping_rounds = 20)