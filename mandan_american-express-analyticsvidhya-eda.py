import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# Loading the data



train_df = pd.read_csv("/kaggle/input/train_AUpWtIz/train.csv")

item_df = pd.read_csv("/kaggle/input/train_AUpWtIz/item_data.csv")

transactions_df = pd.read_csv("/kaggle/input/train_AUpWtIz/customer_transaction_data.csv")

campaign_df = pd.read_csv("/kaggle/input/train_AUpWtIz/campaign_data.csv")

demographics_df = pd.read_csv("/kaggle/input/train_AUpWtIz/customer_demographics.csv")
print("Train Shape ", train_df.shape)

print("Unique Campaign ", train_df.campaign_id.nunique())

print("Unique Coupon ", train_df.coupon_id.nunique())

print("Unique Customer ", train_df.customer_id.nunique())

train_df.head()
sns.countplot(train_df.redemption_status)
print("Number of Unique Id ", item_df['item_id'].nunique())

print("Number of Brands ", item_df['brand'].nunique())

item_df.head()
sns.countplot(item_df['brand_type'])
plt.figure(figsize=(10, 5))

ax = sns.countplot(item_df['category'])



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
print("Number of Transactions ", transactions_df.shape[0])

transactions_df.head()
sns.distplot(transactions_df['selling_price'])
# lets check to see how many transactions used the coupon discount



transactions_df['coupon_discount'].value_counts().head()
# lets look at the largest discount



print(transactions_df['coupon_discount'].min())



# lets look at the row for this item



print(transactions_df[transactions_df['coupon_discount'] == transactions_df['coupon_discount'].min()])
print("Number of Campaigns ", campaign_df['campaign_id'].nunique())



campaign_df.head()
sns.countplot(campaign_df['campaign_type'])
print("Number of Users ", demographics_df['customer_id'].nunique())



demographics_df.head()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))



sns.countplot(demographics_df['age_range'], ax=axes[0])

sns.countplot(demographics_df['marital_status'].fillna('null'), ax=axes[1])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))



sns.countplot(demographics_df['rented'].fillna('null'), ax=axes[0])

sns.countplot(demographics_df['family_size'].fillna('null'), ax=axes[1])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))



sns.countplot(demographics_df['no_of_children'].fillna('null'), ax=axes[0])

sns.countplot(demographics_df['income_bracket'].fillna('null'), ax=axes[1])