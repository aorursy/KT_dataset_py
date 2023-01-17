import numpy as np # linear algebra

import pandas as pd # data processing, CSV file 



import seaborn as sns

import matplotlib.pyplot as plt





import matplotlib.gridspec as gridspec

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import gc

gc.enable()



import os

os.chdir('/kaggle/input/ieeecis-fraud-detection') # Set working directory

print(os.listdir('/kaggle/input/ieeecis-fraud-detection'))
%%time

train_transaction = pd.read_csv('train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('test_identity.csv', index_col='TransactionID')

print ("Data is loaded!")
print('train_transaction shape is {}'.format(train_transaction.shape))

print('test_transaction shape is {}'.format(test_transaction.shape))

print('train_identity shape is {}'.format(train_identity.shape))

print('test_identity shape is {}'.format(test_identity.shape))
train_transaction.head()
train_identity.head()
missing_values_count = train_transaction.isnull().sum()

print (missing_values_count[0:10])

total_cells = np.product(train_transaction.shape)

total_missing = missing_values_count.sum()

print ("% of missing data = ",(total_missing/total_cells) * 100)
missing_values_count = train_identity.isnull().sum()

print (missing_values_count[0:10])

total_cells = np.product(train_identity.shape)

total_missing = missing_values_count.sum()

print ("% of missing data = ",(total_missing/total_cells) * 100)
del missing_values_count, total_cells, total_missing

gc.collect()
ax = sns.countplot(y="isFraud", data=train_transaction)

plt.title('Distribution of  isFraud')



total = len(train_transaction['isFraud'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
train_transaction['TransactionDT'].head()
train_transaction['TransactionDT'].shape[0] , train_transaction['TransactionDT'].nunique()
train_transaction['TransactionDT'].value_counts().head(10)
fig, ax = plt.subplots(1, 2, figsize=(18,4))



time_val = train_transaction['TransactionDT'].values



sns.distplot(time_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of TransactionDT', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])



sns.distplot(np.log(time_val), ax=ax[1], color='b')

ax[1].set_title('Distribution of LOG TransactionDT', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



plt.show()
fig, ax = plt.subplots(1, 2, figsize=(18,4))



time_val = train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionDT'].values



sns.distplot(np.log(time_val), ax=ax[0], color='r')

ax[0].set_title('Distribution of LOG TransactionDT, isFraud=1', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



time_val = train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionDT'].values



sns.distplot(np.log(time_val), ax=ax[1], color='b')

ax[1].set_title('Distribution of LOG TransactionDT, isFraud=0', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])





plt.show()
train_transaction['TransactionDT'].plot(kind='hist',

                                        figsize=(15, 5),

                                        label='train',

                                        bins=50,

                                        title='Train vs Test TransactionDT distribution')

test_transaction['TransactionDT'].plot(kind='hist',

                                       label='test',

                                       bins=50)

plt.legend()

plt.show()
del fig, ax, time_val

gc.collect()
train_transaction.head()
i = 'isFraud'

cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]

train_transaction.loc[train_transaction['isFraud'] == 0].set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3), label="isFraud=0")

train_transaction.loc[train_transaction['isFraud'] == 1].set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3), label="isFraud=1")

plt.legend()

plt.show()
c_features = list(train_transaction.columns[16:30])

for i in c_features:

    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]

    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))

    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))

    plt.show()
del cor, c_features

gc.collect()
d_features = list(train_transaction.columns[30:45])



for i in d_features:

    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]

    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))

    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))

    plt.show()
train_transaction[d_features].head()
# Click output to see the number of missing values in each column

missing_values_count = train_transaction[d_features].isnull().sum()

missing_values_count
# how many total missing values do we have?

total_cells = np.product(train_transaction[d_features].shape)

total_missing = missing_values_count.sum()

# percent of data that is missing

(total_missing/total_cells) * 100
for i in d_features:

    cor_tr = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i].fillna(-1))[0,1]

    cor_te = np.corrcoef(test_transaction['TransactionDT'], test_transaction[i].fillna(-1))[0,1]

    train_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))

    test_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))

    plt.show()
del d_features, cor, missing_values_count, total_cells, total_missing

gc.collect()
m_features = list(train_transaction.columns[45:54])

train_transaction[m_features].head()
i = "V150"

cor_tr = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i].fillna(-1))[0,1]

cor_te = np.corrcoef(test_transaction['TransactionDT'], test_transaction[i].fillna(-1))[0,1]

train_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))

test_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))

plt.show()
del cor_tr, cor_te

gc.collect()
train_transaction.loc[:,train_transaction.columns[train_transaction.columns.str.startswith('V')]].isnull().sum()
fig, ax = plt.subplots(1, 2, figsize=(18,4))



time_val = train_transaction['TransactionAmt'].values



sns.distplot(time_val, ax=ax[0], color='r')

ax[0].set_title('Distribution of TransactionAmt', fontsize=14)

ax[1].set_xlim([min(time_val), max(time_val)])



sns.distplot(np.log(time_val), ax=ax[1], color='b')

ax[1].set_title('Distribution of LOG TransactionAmt', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



plt.show()
fig, ax = plt.subplots(1, 2, figsize=(18,4))



time_val = train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionAmt'].values



sns.distplot(np.log(time_val), ax=ax[0], color='r')

ax[0].set_title('Distribution of LOG TransactionAmt, isFraud=1', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



time_val = train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionAmt'].values



sns.distplot(np.log(time_val), ax=ax[1], color='b')

ax[1].set_title('Distribution of LOG TransactionAmt, isFraud=0', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])





plt.show()
del fig, ax, time_val

gc.collect()
plt.figure(figsize=(10, 7))

d_features = list(train_transaction.columns[30:45])

uniques = [len(train_transaction[col].unique()) for col in d_features]

sns.set(font_scale=1.2)

ax = sns.barplot(d_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del d_features, uniques

gc.collect()
plt.figure(figsize=(10, 7))

c_features = list(train_transaction.columns[16:30])

uniques = [len(train_transaction[col].unique()) for col in c_features]

sns.set(font_scale=1.2)

ax = sns.barplot(c_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center")
del c_features, uniques

gc.collect()
plt.figure(figsize=(35, 8))

v_features = list(train_transaction.columns[54:120])

uniques = [len(train_transaction[col].unique()) for col in v_features]

sns.set(font_scale=1.2)

ax = sns.barplot(v_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del v_features, uniques

gc.collect()
plt.figure(figsize=(35, 8))

v_features = list(train_transaction.columns[120:170])

uniques = [len(train_transaction[col].unique()) for col in v_features]

sns.set(font_scale=1.2)

ax = sns.barplot(v_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del v_features, uniques

gc.collect()
plt.figure(figsize=(35, 8))

v_features = list(train_transaction.columns[170:220])

uniques = [len(train_transaction[col].unique()) for col in v_features]

sns.set(font_scale=1.2)

ax = sns.barplot(v_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del v_features, uniques

gc.collect()
plt.figure(figsize=(35, 8))

v_features = list(train_transaction.columns[220:270])

uniques = [len(train_transaction[col].unique()) for col in v_features]

sns.set(font_scale=1.2)

ax = sns.barplot(v_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del v_features, uniques

gc.collect()
plt.figure(figsize=(35, 8))

v_features = list(train_transaction.columns[270:320])

uniques = [len(train_transaction[col].unique()) for col in v_features]

sns.set(font_scale=1.2)

ax = sns.barplot(v_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del v_features, uniques

gc.collect()
plt.figure(figsize=(38, 8))

v_features = list(train_transaction.columns[320:390])

uniques = [len(train_transaction[col].unique()) for col in v_features]

sns.set(font_scale=1.2)

ax = sns.barplot(v_features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del v_features, uniques

gc.collect()
train_identity.head()
plt.figure(figsize=(35, 8))

features = list(train_identity.columns[0:38])

uniques = [len(train_identity[col].unique()) for col in features]

sns.set(font_scale=1.2)

ax = sns.barplot(features, uniques, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 
del features, uniques

gc.collect()
train_transaction.head(6)
train_identity.head(6)
fig, ax = plt.subplots(1, 2, figsize=(20,5))



sns.countplot(x="ProductCD", ax=ax[0], hue = "isFraud", data=train_transaction)

ax[0].set_title('ProductCD train', fontsize=14)

sns.countplot(x="ProductCD", ax=ax[1], data=test_transaction)

ax[1].set_title('ProductCD test', fontsize=14)

plt.show()
ax = sns.countplot(x="DeviceType", data=train_identity)

ax.set_title('DeviceType', fontsize=14)

plt.show()
print ("Unique Devices = ",train_identity['DeviceInfo'].nunique())

train_identity['DeviceInfo'].value_counts().head()
cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']

for i in cards:

    print ("Unique ",i, " = ",train_transaction[i].nunique())
fig, ax = plt.subplots(1, 4, figsize=(25,5))



sns.countplot(x="card4", ax=ax[0], data=train_transaction.loc[train_transaction['isFraud'] == 0])

ax[0].set_title('card4 isFraud=0', fontsize=14)

sns.countplot(x="card4", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])

ax[1].set_title('card4 isFraud=1', fontsize=14)

sns.countplot(x="card6", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])

ax[2].set_title('card6 isFraud=0', fontsize=14)

sns.countplot(x="card6", ax=ax[3], data=train_transaction.loc[train_transaction['isFraud'] == 1])

ax[3].set_title('card6 isFraud=1', fontsize=14)

plt.show()
"emaildomain" in train_transaction.columns, "emaildomain" in train_identity.columns
fig, ax = plt.subplots(1, 3, figsize=(32,10))



sns.countplot(y="P_emaildomain", ax=ax[0], data=train_transaction)

ax[0].set_title('P_emaildomain', fontsize=14)

sns.countplot(y="P_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])

ax[1].set_title('P_emaildomain isFraud = 1', fontsize=14)

sns.countplot(y="P_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])

ax[2].set_title('P_emaildomain isFraud = 0', fontsize=14)

plt.show()
fig, ax = plt.subplots(1, 3, figsize=(32,10))



sns.countplot(y="R_emaildomain", ax=ax[0], data=train_transaction)

ax[0].set_title('R_emaildomain', fontsize=14)

sns.countplot(y="R_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])

ax[1].set_title('R_emaildomain isFraud = 1', fontsize=14)

sns.countplot(y="R_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])

ax[2].set_title('R_emaildomain isFraud = 0', fontsize=14)

plt.show()
del fig, ax

gc.collect()