import os



def list_all_files_in(dirpath):

    for dirname, _, filenames in os.walk(dirpath):

        for filename in filenames:

            print(os.path.join(dirname, filename))



list_all_files_in('../input')
# Dataframes

import pandas as pd



# Linear algebra

import numpy as np



# Implicit Feedback Recommendation

import implicit



# Sparse matrix

from scipy import sparse



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Displaying stuff

from IPython.display import display



# ZIP I/O

import zipfile



# Paths

from pathlib import Path



# Timing & Timestamps

import time

import datetime



# Serialize Python objects

import pickle



# Iteration

import itertools



# Progress bar

from tqdm.notebook import tqdm

tqdm.pandas()



# Disable warnings

import warnings; warnings.simplefilter('ignore')



color = sns.color_palette()

sns.set_style('white')
# User Master

df_user_list = pd.read_csv('../input/coupon-purchase-prediction-translated/user_list_translated.csv')



# Coupon Master

df_coupon_list_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_list_train_translated.csv')

df_coupon_list_test = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_list_test_translated.csv')



# Coupon Listing Area

df_coupon_area_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_area_train_translated.csv')

df_coupon_area_test = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_area_test_translated.csv')



# Purchase Logs

df_coupon_detail_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_detail_train_translated.csv')



# View Logs

df_coupon_visit_train = pd.read_csv('../input/coupon-purchase-prediction-translated/coupon_visit_train_translated.csv')



# # Unimportant

# df_prefecture_locations = pd.read_csv('../input/coupon-purchase-prediction-translated/prefecture_locations_translated.csv')

# df_submission = pd.read_csv('../input/coupon-purchase-prediction-translated/sample_submission.csv')
def empty_field_count(df_name, df):

    print(f'{df_name} empty field count')

    print(df.isna().sum())

    print('')
empty_field_count('df_coupon_list_train', df_coupon_list_train)

empty_field_count('df_coupon_list_test', df_coupon_list_test)



empty_field_count('df_coupon_area_train', df_coupon_area_train)

empty_field_count('df_coupon_area_test', df_coupon_area_test)



empty_field_count('df_coupon_detail_train', df_coupon_detail_train)

empty_field_count('df_coupon_visit_train', df_coupon_visit_train)



empty_field_count('df_user_list', df_user_list)
# coupon_list_train

coupon_list_ts_cols = ['DISPFROM', 'DISPEND', 'VALIDFROM', 'VALIDEND']



for column in coupon_list_ts_cols:

    df_coupon_list_train[column] = pd.to_datetime(df_coupon_list_train[column])

    df_coupon_list_test[column] = pd.to_datetime(df_coupon_list_test[column])



df_coupon_visit_train['I_DATE'] = pd.to_datetime(df_coupon_visit_train['I_DATE'])



df_user_list['REG_DATE'] = pd.to_datetime(df_user_list['REG_DATE'])

df_user_list['WITHDRAW_DATE'] = pd.to_datetime(df_user_list['WITHDRAW_DATE'])
# df_coupon_visit_train.rename(columns={'VIEW_COUPON_ID_hash':'COUPON_ID_hash'}, inplace=True)

# df_coupon_list_train.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)

# df_coupon_list_test.rename(columns={'large_area_name':'LARGE_AREA_NAME', 'ken_name':'PREF_NAME', 'small_area_name':'SMALL_AREA_NAME'},inplace=True)
# Impute user prefecture with mode

# df_user_list['PREF_NAME'].fillna(df_user_list['PREF_NAME'].value_counts().index[0], inplace=True)

# df_coupon_list_train.fillna(1, inplace=True)

# df_coupon_list_test.fillna(1, inplace=True)



# coupon_fillnat_cols = [('VALIDFROM', 'DISPFROM'), ('VALIDEND', 'DISPEND'), ('VALIDPERIOD', 'DISPPERIOD')]



# for col1, col2 in coupon_fillnat_cols:

#     df_coupon_list_train[col1].fillna(df_coupon_list_train[col2], inplace=True)

#     df_coupon_list_test[col1].fillna(df_coupon_list_test[col2], inplace=True)



# df_user_list['WITHDRAW_DATE'].fillna(pd.Timestamp.max, inplace=True)
plt.figure(figsize=(16, 9))

coupons_by_genre = df_coupon_list_train['GENRE_NAME'].value_counts().reset_index(name='count').rename(columns={'index': 'GENRE_NAME'})

sns.barplot(x='GENRE_NAME', y='count', data=coupons_by_genre)

plt.title('Purchased coupons by genre')

plt.xticks(rotation=90)

plt.show()

# df_coupon_list_train['GENRE_NAME'].value_counts().plot(kind='bar')
plt.figure(figsize=(16, 9))

coupons_by_area = df_coupon_list_train['small_area_name'].value_counts().reset_index(name='count').rename(columns={'index': 'small_area_name'})

sns.barplot(x='small_area_name', y='count', data=coupons_by_area)

plt.title('Purchased coupons by shop area')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(16, 9))

sns.distplot(a=df_coupon_list_train['DISCOUNT_PRICE'], kde=True)

plt.title('Coupon distribution by discount rate (train)')

plt.show()
plt.figure(figsize=(16, 9))

sns.distplot(a=df_coupon_list_test['DISCOUNT_PRICE'], kde=True)

plt.title('Coupon distribution by discount rate (test)')

plt.show()
plt.figure(figsize=(16, 9))

sns.barplot(x=df_user_list['PREF_NAME'].value_counts().index, y=df_user_list['PREF_NAME'].value_counts().values)

plt.title('Users by prefecture')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(16, 9))

coupon_list_train = df_coupon_list_train.set_index('COUPON_ID_hash')

sns.boxplot(y='GENRE_NAME', x='PRICE_RATE', data=coupon_list_train)

plt.title('Price rate by genre')

plt.show()
coupon_visit_train = pd.merge(df_coupon_visit_train, df_user_list, how='left', on='USER_ID_hash').rename(columns={'VIEW_COUPON_ID_hash': 'COUPON_ID_hash'})

coupon_visit_train = coupon_visit_train.join(coupon_list_train, on='COUPON_ID_hash', rsuffix='_coupon')



plt.figure(figsize=(16, 9))

sns.factorplot(x='AGE', y='GENRE_NAME', hue='SEX_ID', kind='violin', data=coupon_visit_train[coupon_visit_train.PURCHASE_FLG==1], orient='h', size=8, scale='count', split=True, cut=0)

plt.title('Age distribution by genre name and gender')

plt.show()
date_indexed_visit = coupon_visit_train.set_index('I_DATE')
fig, ax1 = plt.subplots(figsize=(16, 9))

weekly_purchases = date_indexed_visit.PURCHASE_FLG.resample('W').sum().rename('total').reset_index()

ax1 = sns.lineplot(data=weekly_purchases, x='I_DATE', y='total')

ax2 = ax1.twinx()

weekly_views = date_indexed_visit.PURCHASE_FLG.resample('W').size().reset_index(name='count')

ax2 = sns.lineplot(data=weekly_views, x='I_DATE', y='count', color='orange')

plt.title("Weekly purchases vs views")

plt.show()