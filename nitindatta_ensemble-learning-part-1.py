from IPython.display import Image

import os

Image("../input/ensemble-learning-pic/EL.png")
import pandas as pd

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import datetime

import warnings

import eli5

from eli5.sklearn import PermutationImportance



%matplotlib inline

sns.set(style="darkgrid")

warnings.filterwarnings("ignore")

test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

sales = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates=['date'],dtype={'date': 'str'})
# Concatenating item_categories, items, shops and sales dataframes as train

df = sales.join(items, on='item_id',rsuffix='_')

df = df.join(shops, on='shop_id', rsuffix='_')

df = df.join(item_categories, on='item_category_id', rsuffix='_')
def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df



df = downcast_dtypes(df)

print(df.info())
df.head().T
df.dtypes
print('Dataframe shape :',df.shape)
test_shop_ids = test['shop_id'].unique()

test_item_ids = test['item_id'].unique()

# Only shops that exist in test set.

leak_df = df[df['shop_id'].isin(test_shop_ids)]

# Only items that exist in test set.

leak_df = leak_df[leak_df['item_id'].isin(test_item_ids)]

print('Data set size before leaking:', df.shape[0])

print('Data set size after leaking:', leak_df.shape[0])

df = leak_df
print(df.isnull().sum())

print('\nNo null records')
# We will drop all the strings (object type) and item_category_id as we will not use them.

df.drop(['item_name','shop_name','item_category_name','item_category_id'],axis=1,inplace=True)
print('Is column \'shop_id\' equal to \'shop_id_\' :',df['shop_id'].equals(df['shop_id_']),'\n')

print('Is column \'item_id\' equal to \'item_id_\' :',df['item_id'].equals(df['item_id_']),'\n')

print('\nAll are same so we will drop the duplicates')

df.drop(['shop_id_','item_id_'],axis=1,inplace=True)
df = df[df['item_price']>0]

# Dropped row where item_price is less than 0 
df = df.sort_values('date').groupby(['date_block_num', 'shop_id','item_id'], as_index=False)

df = df.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})

# Rename features.

df.columns = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'mitem_price', 'item_cnt', 'mitem_cnt', 'transactions']
df.count()
df['year'] = df['date_block_num'].apply(lambda x: ((x//12) + 2013))

df['month'] = df['date_block_num'].apply(lambda x: (x % 12))
plt.figure(figsize=(22,8))

plt.subplot(2, 1, 1)

sns.boxplot(x=df['item_cnt'])

plt.subplot(2, 1, 2)

sns.boxplot(x=df['item_price'])
df = df.query('item_cnt >= 0 and item_cnt <= 1500 and item_price < 400000')
df['cnt_m'] = df.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(-1)
df.head()
df.describe().T
corr = df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(9, 7))

    ax = sns.heatmap(corr,mask=mask,square=True,annot=True,fmt='0.2f',linewidths=.8,cmap="YlGnBu")
fig = sns.jointplot(x='item_price',y='item_id',data=df,

                   joint_kws={'alpha':0.2,'color':'orange'},

                   marginal_kws={'color':'red'})
plt.figure(figsize=(20,6)) 

sns.countplot(df['shop_id'])
plt.figure(figsize=(20,6)) 

sns.barplot(x=df['shop_id'],y=df['item_cnt'],palette='viridis')
item_cat_price = df.groupby(['item_id']).sum()['item_price']

plt.figure(figsize=(18,6))

item_cat_price.plot(color ='red')
ts = time.time()

shop_ids = df['shop_id'].unique()

item_ids = df['item_id'].unique()

empty_df = []

for i in range(34):

    for shop in shop_ids:

        for item in item_ids:

            empty_df.append([i, shop, item])

    

empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_id'])

print(time.time()-ts)
# Merge the train set with the complete set (missing records will be filled with 0).

df = pd.merge(empty_df, df, on=['date_block_num','shop_id','item_id'], how='left')

df.fillna(0, inplace=True)
train_set = df.query('date_block_num >= 0 and date_block_num < 26').copy()

validation_set = df.query('date_block_num >= 26 and date_block_num < 33').copy()

test_set = df.query('date_block_num == 33').copy()



print('Train set records:', train_set.shape[0])

print('Validation set records:', validation_set.shape[0])

print('Test set records:', test_set.shape[0])



print('Percent of train_set:',(train_set.shape[0]/df.shape[0])*100,'%')

print('Percent of validation_set:',(validation_set.shape[0]/df.shape[0])*100,'%')

print('Percent of test_set:',(test_set.shape[0]/df.shape[0])*100,'%')
train_set.dropna(subset=['cnt_m'], inplace=True)

validation_set.dropna(subset=['cnt_m'], inplace=True)
# Creating training and validation sets

x_train = train_set.drop(['cnt_m','date_block_num'],axis=1)

y_train = train_set['cnt_m'].astype(int)



x_val = validation_set.drop(['cnt_m','date_block_num'],axis=1)

y_val = validation_set['cnt_m'].astype(int)
latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')

x_test = pd.merge(test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])

x_test['year'] = 2015

x_test['month'] = 9

x_test.drop('cnt_m', axis=1, inplace=True)

x_test = x_test[x_train.columns]
ts=time.time()

sets = [x_train, x_val, x_test]

for dataset in sets:

    for shop_id in dataset['shop_id'].unique():

        for column in dataset.columns:

            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()

            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median

            

# Fill remaining missing values on test set with mean.

x_test.fillna(x_test.mean(), inplace=True)

print(time.time()-ts)
x_test.head()
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from scipy import stats
# These will be our base models

m1 = LinearRegression()

m2 = DecisionTreeRegressor()

m3 = RandomForestRegressor(n_estimators=10)
ts = time.time()

from sklearn.ensemble import VotingRegressor

model = VotingRegressor([('lr', m1), ('dt', m2),('rf', m3)])

model.fit(x_train, y_train)

train_pred = model.predict(x_train)

val_pred = model.predict(x_val)

print('Total time taken :',time.time()-ts) 
print('Train rmse:', np.sqrt(mean_squared_error(y_train, train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(y_val, val_pred)))
perm = PermutationImportance(model, random_state=1).fit(x_val, y_val)

eli5.show_weights(perm, feature_names = x_val.columns.tolist())
ts = time.time()

m1.fit(x_train, y_train)

m2.fit(x_train, y_train)

m3.fit(x_train,y_train)



avg_train_pred1 = m1.predict(x_train)

avg_train_pred2 = m2.predict(x_train)

avg_train_pred3 = m3.predict(x_train)



avg_pred1 = m1.predict(x_val)

avg_pred2 = m2.predict(x_val)

avg_pred3 = m3.predict(x_val)



train_pred_avg = (avg_train_pred1+avg_train_pred2+avg_train_pred3)/3

val_pred_avg = (avg_pred1+avg_pred2+avg_pred3)/3



print('Total time taken: ',time.time()-ts)
print('Train rmse:', np.sqrt(mean_squared_error(y_train, train_pred_avg)))

print('Validation rmse:', np.sqrt(mean_squared_error(y_val, val_pred_avg)))
ts = time.time()

m1.fit(x_train, y_train)

m2.fit(x_train, y_train)

m3.fit(x_train,y_train)



wavg_train_pred1 = m1.predict(x_train)

wavg_train_pred2 = m2.predict(x_train)

wavg_train_pred3 = m3.predict(x_train)



print('M1_train:',np.sqrt(mean_squared_error(y_train, wavg_train_pred1)))

print('M2_train:',np.sqrt(mean_squared_error(y_train, wavg_train_pred2)))

print('M3_train:',np.sqrt(mean_squared_error(y_train, wavg_train_pred3)))



wavg_pred1 = m1.predict(x_val)

wavg_pred2 = m2.predict(x_val)

wavg_pred3 = m3.predict(x_val)



print('\nM1_validation:',np.sqrt(mean_squared_error(y_val, wavg_pred1)))

print('M2_validation:',np.sqrt(mean_squared_error(y_val, wavg_pred2)))

print('M3_validation:',np.sqrt(mean_squared_error(y_val, wavg_pred3)))



print('\nTotal time taken: ',time.time()-ts)
final_val_pred = 0.3 * wavg_pred1 + 0.2 * wavg_pred2 + 0.5 * wavg_pred3

print('Weighted Average:',np.sqrt(mean_squared_error(y_val, final_val_pred)))
train_set.to_csv('/kaggle/working/train_set.csv',index=False)

validation_set.to_csv('/kaggle/working/validation_set.csv',index=False)

test_set.to_csv('/kaggle/working/test_set.csv',index=False)