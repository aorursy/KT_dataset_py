# Importing all the required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading the all the data set

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')



test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
# Data Shape

print('Data Shape of train identity :', train_identity.shape)

print('Data Shape of train transaction :', train_trans.shape)



print('Data Shape of test identity :', test_identity.shape)

print('Data Shape of test transaction :', test_trans.shape)
# Data Overview

print("Train Identity")

train_identity.head(10).T
# Data Overview

print("Train Transaction")

train_trans.head(10).T
# Merging the dataset

train_merged = pd.merge(train_trans, train_identity, on = 'TransactionID')

test_merged = pd.merge(test_trans, test_identity, on = 'TransactionID')
del train_identity, train_trans, test_identity, test_trans
# Target in Analysis

plt.subplots(figsize = (6,6))

train_merged['isFraud'].value_counts().plot('pie')

plt.show()
# Missing Values

missing_values = ((train_merged.isnull().sum()/len(train_merged)).sort_values(ascending = False))*100

print("Missing Values")

print(missing_values)
# Removing all the variables having more than 90% of the missing value

cols_to_remove = missing_values[missing_values>=90].index.tolist()
# Removing above columns from train and test data

train_merged.drop(cols_to_remove, axis =1, inplace = True)

test_merged.drop(cols_to_remove, axis =1, inplace = True)
train_merged.head(10).T
# Splitting data into categorical and numerical

categorial_data = train_merged.select_dtypes(include = ['object'])

numerical_data = train_merged.select_dtypes(exclude = ['object'])
categorial_data.head(10).T
categorial_data.describe().T
# ProductCD Analysis

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot('ProductCD', data = categorial_data, ax=ax[0])

sns.countplot(categorial_data['ProductCD'], hue = numerical_data['isFraud'], ax=ax[1])

plt.subplots_adjust(wspace = 0.5)

plt.show()
# Card4 Analysis

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot('card4', data = categorial_data, ax=ax[0])

sns.countplot(categorial_data['card4'], hue = numerical_data['isFraud'], ax=ax[1])

plt.subplots_adjust(wspace = 0.5)

plt.show()
# Card6 Analysis

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot('card6', data = categorial_data, ax=ax[0])

sns.countplot(categorial_data['card6'], hue = numerical_data['isFraud'], ax=ax[1])

plt.subplots_adjust(wspace = 0.5)

plt.show()
# P_emaildomain

plt.figure(figsize=(20, 6))

sns.countplot('P_emaildomain', data = categorial_data)

plt.xticks(rotation=90)

plt.show()
temp = pd.DataFrame(categorial_data['P_emaildomain'].value_counts())

temp['%cent'] = temp['P_emaildomain']/temp['P_emaildomain'].sum()*100

temp['others'] = np.where(temp['%cent'] > 1, temp.index, 'others')

p_email_mapping = temp['others'].to_dict()

categorial_data['P_emaildomain_new'] = categorial_data['P_emaildomain'].map(p_email_mapping)
# P_email and P_email_new Analysis

fig, ax = plt.subplots(1, 2, figsize=(20, 6))

categorial_data['P_emaildomain'].value_counts().plot(kind = 'bar', ax = ax[0])

categorial_data['P_emaildomain_new'].value_counts().plot(kind = 'bar', ax = ax[1])

plt.show()
# R_emaildomain

plt.figure(figsize=(20, 6))

categorial_data['R_emaildomain'].value_counts().plot(kind = 'bar')

plt.show()
# M4

# R_emaildomain

plt.figure(figsize=(20, 6))

categorial_data['M4'].value_counts().plot(kind = 'pie')

plt.show()
# Taking all the variable started with id

id_cols = categorial_data.columns[categorial_data.columns.str.startswith('id')].tolist()

# Removing id_30, id_31, id_33: Want to look at them separately due to high number of levels in it

cols_to_remove = ['id_30', 'id_31', 'id_33']

for i in cols_to_remove:

    id_cols.remove(i)
fig, ax = plt.subplots(5, 2, figsize = (15, 20),

                       gridspec_kw={'hspace' : 0.5, 'wspace':0.2})

ax = np.reshape(ax, (10))



for col, axis in zip(id_cols, ax):

    sns.countplot(categorial_data[col], ax = axis)
#cols_to_remove = ['id_30', 'id_31', 'id_33']

# id_30

plt.figure(figsize=(20, 6))

categorial_data['id_30'].value_counts().plot(kind = 'bar')

plt.show()