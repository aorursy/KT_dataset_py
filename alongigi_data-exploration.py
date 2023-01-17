import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
path = '../input/'

train_transaction = pd.read_csv(path + 'train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv(path + 'test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv(path + 'train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv(path + 'test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv(path + 'sample_submission.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



y_train = train['isFraud'].copy()

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()



del train, test, train_transaction, train_identity, test_transaction, test_identity
X_train.shape[0]
X_test.shape[0]
X_train.shape[1]
type_count = X_train.dtypes.value_counts()

sns.barplot(type_count.index.astype('str'), type_count.values, alpha=0.8)

plt.title('Dataset columns types')

plt.ylabel('Number of Columns', fontsize=12)

plt.xlabel('Type', fontsize=12)

plt.show()
X_train.isnull().sum()
X_train.select_dtypes(['object']).columns
f, axes = plt.subplots(6, 4, figsize=(24, 30))



object_type_columns = ['ProductCD','card4','card6','M2','M3','M4','M5','M6','M7','M8','M9','id_12','id_15','id_16',

                       'id_23','id_27','id_28','id_29','id_34','id_35','id_36','id_37','id_38','DeviceType']



row, col, index = 0, 0, 0



for i in range(6):

    for j in range(4):

        sns.countplot(x=object_type_columns[index], data=X_train, hue=y_train, ax=axes[i][j])

        index = index + 1



plt.tight_layout()
product_count = X_train.ProductCD.value_counts()

sns.barplot(product_count.index, product_count.values, alpha=0.8)

plt.title('ProductCD frequency')

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('ProductCD', fontsize=12)

plt.show()
device_count = X_train.DeviceInfo.value_counts().head()

sns.barplot(device_count.index, device_count.values, alpha=0.8)

plt.title('Most common DeviceInfo frequency')

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('DeviceInfo', fontsize=12)

plt.show()
labels = 'isFraud', 'isNotFraud'

sizes = [len(y_train[y_train==1]),len(y_train[y_train==0])]

explode = (0.2, 0)

colors =  ['red','green']



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, colors = colors, autopct='%1.2f%%')

ax1.axis('equal')



plt.show()