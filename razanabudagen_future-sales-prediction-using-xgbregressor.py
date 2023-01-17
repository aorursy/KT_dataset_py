import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')



train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')



sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
#Reading the Data



print('item_categories')

display(item_categories.head())



print('items')

display(items.head())



print('shops')

display(shops.head())



print('train')

display(train.head())



print('test')

display(test.head())



print('sample_submission')

display(sample_submission.head())
train.info()
#Check Missing Values



print('train')

display(train.isnull().sum())



print('test')

display(test.isnull().sum())
print('train')

display(train.describe(include='all'))



print('test')

display(test.describe(include='all'))
#drop duplicates



subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']

print(train.duplicated(subset=subset).value_counts())

train.drop_duplicates(subset=subset, inplace=True)
#Check negative values in item_price



train[train['item_price'] < 0]
#drop negative value in item_price



train = train[train['item_price'] > 0]





train = train[train['item_cnt_day'] > 0]
sns.boxplot(train['item_price']);
sns.boxplot(train['item_cnt_day']);
def drop_outliers(df, feature, percentile_high = .99):



    #train size before dropping values

    shape_init = df.shape[0]



    max_value = df[feature].quantile(percentile_high)



    print('dropping outliers...')

    df = df[df[feature] < max_value]

    

    print(str(shape_init - df.shape[0]) + ' ' + feature + 

          ' values over ' + str(max_value) + ' have been removed' )

    

    return df
#drop outliers in item_price feature



train = drop_outliers(train, 'item_price')
#drop outliers in item_cnt_day



train = drop_outliers(train, 'item_cnt_day')
prices_shop_df = train[['shop_id','item_id','item_price']]

prices_shop_df = prices_shop_df.groupby(['shop_id','item_id']).apply(lambda df: df['item_price'][-2:].mean())

prices_shop_df = prices_shop_df.to_frame(name = 'item_price')



prices_shop_df
test = pd.merge(test, prices_shop_df, how='left', left_on=['shop_id','item_id'], right_on=['shop_id','item_id'])



test.head()
#check for missing values



test['item_price'].isnull().sum()
#split content in date into month and year

train['month'] = [date.split('.')[1] for date in train['date']]

train['year'] = [date.split('.')[2] for date in train['date']]



#drop date and date_block_num features

train.drop(['date','date_block_num'], axis=1, inplace=True)



#create month and year features fot test dataset

test['month'] = '11'

test['year'] = '2015'
#change item_cnt_day into item_cnt_month

train_monthly = train.groupby(['year','month','shop_id','item_id'], as_index=False)[['item_cnt_day']].sum()

train_monthly.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)



train_monthly = pd.merge(train_monthly, prices_shop_df, how='left', left_on=['shop_id','item_id'], right_on=['shop_id','item_id'])



train_monthly.head()
train = train_monthly
#REINDEX TEST DATASET



test = test.reindex(columns=['ID','year','month','shop_id','item_id','item_price'])



test.head()
#EXPLORING ITEMS CATEGORY DATASET



#extract main categories

item_categories['main_category'] = [x.split(' - ')[0] for x in item_categories['item_category_name']]



sub_categories = []

for i in range(len(item_categories)):

    try:

        sub_categories.append(item_categories['item_category_name'][i].split(' - ')[1])

        

    except IndexError as e:

        sub_categories.append('None')



item_categories['sub_category'] = sub_categories



item_categories.drop(['item_category_name'], axis=1, inplace=True)



item_categories.head()
#EXPLORING ITEMS DATASET



#merge with item_categories

items = pd.merge(items, item_categories, how='left')



items.drop(['item_name','item_category_id'], axis=1, inplace=True)



items.head()
#merge to train and test datasets



train = pd.merge(train, items, how='left')

test = pd.merge(test, items, how='left')
#EXPLORING SHOPS DATASET





from string import punctuation



# replace all the punctuation in the shop_name columns

shops["shop_name_cleaned"] = shops["shop_name"].apply(lambda s: "".join([x for x in s if x not in punctuation]))



shops["shop_city"] = shops["shop_name_cleaned"].apply(lambda s: s.split()[0])



shops["shop_type"] = shops["shop_name_cleaned"].apply(lambda s: s.split()[1])



shops["shop_name"] = shops["shop_name_cleaned"].apply(lambda s: " ".join(s.split()[2:]))



shops.drop(['shop_name_cleaned'], axis=1, inplace=True)



shops.head()
#merge to train and test datasets



train = pd.merge(train, shops, how='left')

test = pd.merge(test, shops, how='left')
print('train')

display(train.head())



print('test')

display(test.head())
#FILL MISSING VALUES IN ITEM_PRICE (by item categories)



#fill missing values with median of each main_category and sub_category

test['item_price'] = test.groupby(['main_category','sub_category'])['item_price'].apply(lambda df: df.fillna(df.median()))



test['item_price'].isnull().sum()
#fill missing values with median of each sub_category

test['item_price'] = test.groupby(['sub_category'])['item_price'].apply(lambda df: df.fillna(df.median()))



test['item_price'].isnull().sum()
test[test['item_price'].isnull()]
#fill missing values with median of main_category and sub_category from train dataset

filler = train[(train['main_category'] == 'PC') & (train['sub_category'] == 'Гарнитуры/Наушники')]['item_price'].median()



test['item_price'].fillna(filler, inplace=True)

test['item_price'].isnull().sum()
train['item_cnt_month'] = train['item_cnt_month'].clip(0,20)
target_array = train['item_cnt_month']

train.drop(['item_cnt_month'], axis=1, inplace=True)



test_id = test['ID']

test.drop(['ID'], axis=1, inplace=True)
train.drop(['shop_id','item_id'], axis=1, inplace=True)

test.drop(['shop_id','item_id'], axis=1, inplace=True)
def downcast_dtypes(df):



    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    

    return df
#reduce memory

downcast_dtypes(train)

downcast_dtypes(test)
train.info()
#check for any missing data

print('missing data in the train dataset : ', train.isnull().any().sum())

print('missing data in the test dataset : ', test.isnull().any().sum())
def normalityTest(data, alpha=0.05):



    from scipy import stats

    

    statistic, p_value = stats.normaltest(data)



    if p_value < alpha:  

        is_normal_dist = False

    else:

        is_normal_dist = True

    

    return is_normal_dist
#check normality of all numericaal features and transform it if not normal distributed



for feature in train.columns:

    if (train[feature].dtype != 'object'):

        if normalityTest(train[feature]) == False:

            train[feature] = np.log1p(train[feature])

            test[feature] = np.log1p(test[feature])
target_array = np.log1p(target_array)
#ENCODING



from sklearn.preprocessing import OrdinalEncoder



enc = OrdinalEncoder()



X = enc.fit_transform(train)

y = target_array



X_predict = enc.fit_transform(test)
#SPLITTING THE DATA IN ORDER TO CREATE THE MODEL



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state = 0)
from xgboost import XGBRegressor



#create a model

model = XGBRegressor()



#fitting

model.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_train, y_train), (X_test, y_test)], 

          verbose=True, early_stopping_rounds = 20)
#calculate Mean Squared Error

from sklearn.metrics import mean_squared_error



print('MSE : ', mean_squared_error(y_test, model.predict(X_test)))
y_predict = model.predict(X_predict)

y_predict = np.expm1(y_predict)
results = pd.DataFrame({'ID': test_id, 'item_cnt_month': y_predict})

results.to_csv('my_submission.csv', index=False)