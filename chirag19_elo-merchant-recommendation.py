import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import datetime as dt



import warnings

warnings.filterwarnings('ignore')
merchants = pd.read_csv('../input/merchants.csv')

historical_transactions = pd.read_csv('../input/historical_transactions.csv')

new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')

train = pd.read_csv('../input/train.csv', parse_dates = ['first_active_month'])

test = pd.read_csv('../input/test.csv', parse_dates = ['first_active_month'])
print("Historical_transactions   :", historical_transactions.shape)

print("Merchants                 :", merchants.shape)

print("New_merchant_transactions :", new_merchant_transactions.shape)

print("Train set                 :", train.shape)

print("Test set                  :", test.shape)
print(pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name = 'train'))
print('target:', pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name = 'train')['Unnamed: 1'][7])
train.head(2)
train.describe()
plt.figure(figsize = (10,6))

sns.distplot(train.target)

plt.show()
plt.figure(figsize = (10,5))

plt.scatter(train.index, np.sort(train.target.values))

plt.xlabel('Index')

plt.ylabel('Loyalty Score')

plt.show()
print((train.target == -33.21928095).sum())

print((train.target == -33.21928095).sum()*100 / len(train.target))
plt.figure(figsize = (14,6))

sns.boxplot('feature_1', 'target', data = train)

plt.title('Feature_1 distribution (loyalty score)')

plt.show()



plt.figure(figsize = (12,6))

sns.boxplot('feature_2', 'target', data = train)

plt.title('Feature_2 distribution (loyalty score)')

plt.show()



plt.figure(figsize = (8,4))

sns.boxplot('feature_3', 'target', data = train)

plt.title('Feature_3 distribution (loyalty score)')

plt.show()
train['new_target'] = 2**train.target
train.new_target.describe()
plt.figure(figsize = (10,6))

sns.distplot(train.new_target)

plt.show()
train.sort_values(by = 'new_target').head(5)
train.sort_values(by = 'new_target').tail(5)
plt.figure(figsize = (14,6))

sns.boxplot('feature_1', 'new_target', data = train)

plt.title('Feature_1 distribution (loyalty score)')

plt.show()



plt.figure(figsize = (12,6))

sns.boxplot('feature_2', 'new_target', data = train)

plt.title('Feature_2 distribution (loyalty score)')

plt.show()



plt.figure(figsize = (8,4))

sns.boxplot('feature_3', 'new_target', data = train)

plt.title('Feature_3 distribution (loyalty score)')

plt.show()
train['year'] = train['first_active_month'].dt.year
plt.figure(figsize = (8,6))

plt.scatter(train.year, train.new_target, alpha = 0.5)

plt.xlabel('First Active Year')

plt.ylabel('Loyalty Score (new_target)')

plt.title('Loyalty Score vs First Active Year')

plt.show()
month_count = train.first_active_month.dt.date.value_counts().sort_index()



plt.figure(figsize = (14,6))

sns.barplot(month_count.index, month_count.values, color = 'r')

plt.xticks(rotation = 'vertical')

plt.title('Train data distribution based on first active month')

plt.show()



plt.figure(figsize = (14,6))

month_count = test.first_active_month.dt.date.value_counts().sort_index()

sns.barplot(month_count.index, month_count.values, color = 'g')

plt.xticks(rotation = 'vertical')

plt.title('Test data distribution based on first active month')

plt.show()
print(pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name = 'new_merchant_period'))
new_merchant_transactions.head()
hist_new = pd.concat((historical_transactions, new_merchant_transactions), ignore_index = True)
hist_new.shape
hist_new.purchase_amount.describe()
hist_new['new_amount'] = hist_new.purchase_amount - hist_new.purchase_amount.min()
hist_new.new_amount.describe()
np.sort(hist_new.new_amount.unique())[:10]
np.diff(np.sort(hist_new.new_amount.unique()))
hist_new_sorted = hist_new.groupby('new_amount').new_amount.first().to_frame().reset_index(drop=True)

hist_new_sorted['delta'] = hist_new_sorted.new_amount.diff()

hist_new_sorted[hist_new_sorted.delta >= 2e-5].head()
hist_new_sorted[1:52623].delta.mean()
hist_new['new_amount'] = np.round(hist_new['new_amount'] / (100 * hist_new_sorted[1:52623].delta.mean()), 2)
hist_new.new_amount.value_counts().head()
historical_transactions = hist_new[:29112361]

new_merchant_transactions = hist_new[29112361:]
historical_transactions.shape, new_merchant_transactions.shape