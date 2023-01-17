import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc



#sns color plalette.

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

colour = sns.color_palette(flatui)



#Path for the directory

PATH = '../input/amexpert/'
#Reading the dataset.

train = pd.read_csv(f'{PATH}train.csv')

test = pd.read_csv(f'{PATH}test_QyjYwdj.csv')

campaign_data = pd.read_csv(f'{PATH}campaign_data.csv')

coupon_item = pd.read_csv(f'{PATH}coupon_item_mapping.csv')

customer_trans = pd.read_csv(f'{PATH}customer_transaction_data.csv')

item_data = pd.read_csv(f'{PATH}item_data.csv')

customer_demo = pd.read_csv(f'{PATH}customer_demographics.csv')
#train first 5 observations

train.head()
#shape of the dataset.

print(f'Train shape: {train.shape}')
#function to get the number of unique items in a feature.

def get_unique(df, col=None):

    assert col == None or type(col) == list, 'Either None or List.'

    if col:

        for i in col:

            print(f'Unique items in {i} are {df[i].unique().shape[0]}')

    else:

        for col in df.columns:

            print(f'Unique items in {col} are {df[col].unique().shape[0]}')
#getting unique items in the data.

get_unique(train, col=['id', 'campaign_id', 'coupon_id', 'customer_id'])
#Distribution of the redemption_status.

plt.figure(figsize=(7,5))

sns.countplot(train.redemption_status, color=colour[0])
#value count

train.redemption_status.value_counts()
#imbalance ratio:

print('*'*70)

print(train.redemption_status.value_counts(normalize=True))

print('*'*70, '\n')

print(f'Imbalance ratio: {train.redemption_status.value_counts()[1] / train.redemption_status.value_counts()[0]}')
#Campaign data.

campaign_data.head()
get_unique(campaign_data)
#Campaign type

plt.figure(figsize=(7,5))

sns.countplot(campaign_data.campaign_type, color=colour[1])
#Start date.

campaign_data.start_date.describe()
#End date.

campaign_data.end_date.describe()
#shape of the data.

campaign_data.shape
#coupon item mapping

coupon_item.head()
#unique data

get_unique(coupon_item)
#items data

item_data.head()
#shape:

item_data.shape
#getting unique items from this data.

get_unique(item_data)
#plotting brand_type

plt.figure(figsize=(7,5))

sns.countplot(item_data.brand_type, color=colour[2])
#brand category.

plt.figure(figsize=(17,5))

sns.countplot(item_data.category, color=colour[3])

plt.xticks(rotation=90);
#lets merge the items data with coupon_item mapping.

coupon_item = coupon_item.merge(item_data, on='item_id', how='left')

coupon_item.head()
#shape

coupon_item.shape
#null values.

coupon_item.isnull().sum()
#category on which most coupons are given.

group1 = coupon_item.groupby('category')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)

plt.figure(figsize=(17,5))

sns.barplot(group1.category, group1.count_coupons, color=colour[4])

plt.xticks(rotation=90);
#brand on which most coupons are given.

%time group2 = coupon_item.groupby('brand')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)

plt.figure(figsize=(17,5))

sns.barplot('brand', 'count_coupons',

            data=group2.head(10), #considering only top 10

            color=colour[5])

plt.xticks(rotation=90);
#items on which most coupons are given.

%time group3 = coupon_item.groupby('item_id')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)

plt.figure(figsize=(17,5))

sns.barplot('item_id', 'count_coupons', 

            data=group3.head(10), #considering only top 10

            color=colour[0])

plt.xticks(rotation=90);
#brand_type on which most coupons are given.

%time group4 = coupon_item.groupby('brand_type')['coupon_id'].agg('count').to_frame('count_coupons').reset_index().sort_values('count_coupons', ascending=False)

plt.figure(figsize=(7,5))

sns.barplot('brand_type', 'count_coupons', 

            data=group4,

            color=colour[1])

# plt.xticks(rotation=90);
#customer demographics.

customer_demo.head()
#shape

customer_demo.shape
#unique_items.

get_unique(customer_demo)
#frequency of age range

plt.figure(figsize=(8,5))

sns.countplot(customer_demo.age_range, color=colour[2])
#marital status.

customer_demo.marital_status.value_counts(dropna=False)
#rented

plt.figure(figsize=(7, 5))

sns.countplot(customer_demo.rented, color=colour[3])
#family size.

#rented

plt.figure(figsize=(7, 5))

sns.countplot(customer_demo.family_size, color=colour[4])
#number of childrens

#rented

plt.figure(figsize=(7, 5))

sns.countplot(customer_demo.no_of_children, color=colour[4])
#number of childrens

#rented

plt.figure(figsize=(7, 5))

sns.countplot(customer_demo.income_bracket, color=colour[4])
#null values present in this data.

customer_demo.isnull().sum()
#Customer Transaction data.

customer_trans.head()
#number of unique items

get_unique(customer_trans)
#distribution of quantity

sns.distplot(customer_trans.quantity, bins=50);
#description

customer_trans.quantity.describe()
#subsetting.

high_order = customer_trans.loc[customer_trans.quantity >= 50]

high_order.sort_values('quantity', ascending=False, inplace=True)

high_order.head()
#selling price.

customer_trans.selling_price.describe()
#selling price distribution.

sns.distplot(np.log1p(customer_trans.selling_price), bins=20)
#other discount.

customer_trans.other_discount.describe()
#coupon discount.

customer_trans.coupon_discount.describe()
#distribution

sns.distplot(abs(customer_trans.coupon_discount))
#shape of the dataset.

customer_trans.shape