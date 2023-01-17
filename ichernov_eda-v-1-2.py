import os
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
PATH_TO_DATA = "../input/"

train = pd.read_csv(PATH_TO_DATA + 'train.csv.gz', encoding='utf-8')
train_checks = pd.read_csv(PATH_TO_DATA + 'train_checks.csv.gz', encoding='utf-8')
train.head()
train_checks.head()
# Get rid of na
train['name'].fillna('', inplace=True)

# Parse datetime
train = pd.merge(train,
                 train_checks[['check_id', 'datetime']],
                 how='left',
                 on='check_id')
train['datetime'] = pd.to_datetime(train['datetime'])
train['purchase_hour'] = train['datetime'].dt.hour
train['purchase_weekday'] = train['datetime'].dt.weekday
y_le = LabelEncoder()
y_train_complete = y_le.fit_transform(train['category'])
train['category_encoded'] = y_train_complete.copy()
print(len(train[train['price'] < 1000]), len(train[train['price'] >= 1000]))
# For now let's ignore items with the price of
# 1k RUB and greater in order to create
# a clearer picture of the distribution.

sns.distplot(train[train['price'] < 1000]['price'], bins=50)
plt.xlabel('Item\'s Price');
plt.figure(figsize=(20, 12))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.title('Category "' + y_le.classes_[i] + '"')
    sns.distplot(train[train['category_encoded']==i]['price'], bins=50)
    plt.xlabel('Items\' Price')

plt.tight_layout();
plt.figure(figsize=(20, 7))
sns.boxplot(train['category'], train['price'])
plt.yscale('log')
plt.xticks(rotation=90)
plt.ylabel('Items\' Price')
plt.xlabel('Category Title');
hours = range(24)
weekday_numbers = range(7)
purchase_counts = []

for i in range(25):
    purchase_count = np.zeros((len(weekday_numbers), len(hours)))
    for w, h in itertools.product(weekday_numbers, hours):
        purchase_count[w][h] = train[(train['category_encoded']==i)&\
                                     (train['purchase_weekday']==w)&\
                                     (train['purchase_hour']==h)]['check_id'].count()
    
    purchase_counts.append(purchase_count)
plt.figure(figsize=(20, 24))

for i in range(25):
    plt.subplot(9, 3, i + 1)
    plt.title('Category "' + y_le.classes_[i] + '"')
    sns.heatmap(purchase_counts[i], square=True)
    plt.xlabel('Hour')
    plt.ylabel('Weekday Number')

plt.tight_layout();
plt.figure(figsize=(20, 7))
sns.stripplot(train['datetime'], train['category'], size=3, jitter=True)
plt.xlabel('Year and Month')
plt.ylabel('Category Title');
plt.figure(figsize=(20, 6))

# PRICE
plt.subplot(141)
sns.boxplot(train['price'], orient='v')
plt.ylabel('Price')
plt.subplot(142)
sns.boxplot(train['price'], orient='v')
plt.yscale('log')
plt.ylabel('Price (log)')

# COUNT
plt.subplot(143)
sns.boxplot(train['count'], orient='v')
plt.ylabel('Count')
plt.subplot(144)
sns.boxplot(train['count'], orient='v')
plt.yscale('log')
plt.ylabel('Count (log)')

plt.tight_layout();
plt.figure(figsize=(20, 14))
plt.subplot(211)
sns.boxplot(train['price'], train['category'])
plt.ylabel('Category Title')
plt.xscale('log')
plt.xlabel('Price (log)')

plt.subplot(212)
sns.boxplot(train['count'], train['category'])
plt.ylabel('Category Title')
plt.xscale('log')
plt.xlabel('Count (log)');