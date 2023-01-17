# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.model_selection import train_test_split

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
train_transaction.head()
train_transaction.tail()
train_transaction.info(verbose=True, memory_usage=True, null_counts=True)
train_transaction.describe()
train_transaction.describe(include='O')
#columns with different types



obj_col = train_transaction.select_dtypes(include='O').columns

float_col = train_transaction.select_dtypes(include='float64').columns

int_col = train_transaction.select_dtypes(include='int64').columns

train_transaction[obj_col].head()
train_transaction[obj_col].info()
train_transaction[float_col].head(10)
train_transaction[float_col].info()
np.min(np.min(train_transaction[float_col])), np.max(np.max(train_transaction[float_col]))
np.finfo(np.float64).min, np.finfo(np.float64).max, np.finfo(np.float32).min, np.finfo(np.float32).max  
#Reducing memory usage

#Converting float64 columns to float32

train_transaction[float_col] = train_transaction[float_col].astype('float32')
train_transaction[float_col].info()
train_transaction[int_col].head()
np.min(np.min(train_transaction[int_col])), np.max(np.max(train_transaction[int_col]))
train_transaction[int_col].info()
train_transaction['isFraud'].value_counts()/len(train_transaction)*100
plt.figure(figsize=(2.5,5))

plt.title("Fraud Transaction Distribution")

p1 = sns.countplot(train_transaction['isFraud'], palette = 'plasma')



for p in p1.patches:

        p1.annotate('{:6.2f}%'.format(p.get_height()/len(train_transaction)*100), (p.get_x()+0.1, p.get_height()+50))

        

plt.show()
train_transaction[['TransactionAmt','isFraud']].groupby('isFraud').sum()
train_transaction.dtypes.value_counts()
train_transaction.shape, test_transaction.shape
(train_transaction.memory_usage()/(1024*1024)).sort_values(ascending=False)
train_identity.head()
train_identity.info()
train_identity.shape, test_identity.shape
pd.options.display.max_rows = 999

(np.sum(pd.isnull(train_transaction)).sort_values(ascending=False)/len(train_transaction))*100



(np.sum(pd.isnull(test_transaction)).sort_values(ascending=False)/len(test_transaction))*100

print("Transaction Amounts Quantiles:\n", train_transaction['TransactionAmt'].quantile([.01, .025, 0.1, 0.25, 0.5, .75, 0.9, .975, .990]))

plt.text(25000, 0.004, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(25000, 0.0036, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['TransactionAmt'], color='red')

sns.distplot(test_transaction['TransactionAmt'], color='blue')



plt.show()
train_transaction[['TransactionAmt', 'isFraud']].groupby('isFraud')
plt.text(0, 0.8, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(0, 0.72, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(np.log1p(train_transaction['TransactionAmt']), color='red')

sns.distplot(np.log1p(test_transaction['TransactionAmt']), color='blue')



plt.show()

plt.text(9000, 0.004, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(9000, 0.0036, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['dist1'], color='red')

sns.distplot(test_transaction['dist2'], color='blue')



plt.show()
plt.text(9,0.4, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(9, 0.35, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(np.log1p(train_transaction['dist1']), color='red')

sns.distplot(np.log1p(test_transaction['dist2']), color='blue')



plt.show()
plt.text(4000, 0.01, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(4000, 0.012, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['C1'], color='red')

sns.distplot(test_transaction['C1'], color='blue')



plt.show()
plt.text(8, 2, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(8, 2.5, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(np.log1p(train_transaction['C1']), color='red')

sns.distplot(np.log1p(test_transaction['C1']), color='blue')



plt.show()
plt.text(4000, 0.01, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(4000, 0.012, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['C2'], color='red')

sns.distplot(test_transaction['C2'], color='blue')



plt.show()
plt.text(8, 2, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(8, 2.4, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(np.log1p(train_transaction['C2']), color='red')

sns.distplot(np.log1p(test_transaction['C2']), color='blue')



plt.show()
plt.text(600, 0.03, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(600, 0.035, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['D1'], color='red', hist=False)

sns.distplot(test_transaction['D1'], color='blue', hist=False)



plt.show()
plt.text(6, 0.6, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(6, 0.7, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(np.log1p(train_transaction['D1']), color='red', hist=False)

sns.distplot(np.log1p(test_transaction['D1']), color='blue', hist=False)



plt.show()
plt.text(600, 0.003, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(600, 0.0035, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['D2'], color='red', hist=False)

sns.distplot(test_transaction['D2'], color='blue', hist=False)



plt.show()
plt.text(3, 0.2, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(3, 0.25, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(np.log1p(train_transaction['D2']), color='red', hist=False)

sns.distplot(np.log1p(test_transaction['D2']), color='blue', hist=False)



plt.show()
fig, ax =plt.subplots(1,2)

ax[0].set_title('Train')

ax[1].set_title('Test')



sns.countplot(train_transaction['ProductCD'], ax=ax[0])

sns.countplot(test_transaction['ProductCD'], ax=ax[1])

fig.show()



plt.figure(figsize=(6,6))

plt.title("Transaction Amount Distribuition by ProductCD and Target")

sns.boxenplot(x='ProductCD', y='TransactionAmt',data=train_transaction.loc[train_transaction['TransactionAmt']<2000], hue='isFraud')

plt.show()
ax = sns.countplot(x='ProductCD',data=train_transaction,hue='isFraud', palette='plasma')



plt.show()

print("Card Quantiles:\n", train_transaction[['card1','card2','card3', 'card5']].quantile([.01, .025, 0.1, 0.25, 0.5, .75, 0.9, .975, .990]))

g = sns.FacetGrid(train_transaction[['card1','isFraud']], hue="isFraud", height=5, aspect=2)

g = g.map(sns.distplot, "card1")

plt.legend()

plt.show()
g = sns.FacetGrid(train_transaction[['card2','isFraud']], hue="isFraud", height=5, aspect=2)

g = g.map(sns.distplot, "card2")

plt.legend()

plt.show()
plt.text(6000, 0.003, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(6000, 0.0035, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['card1'], color='red', hist=False)

sns.distplot(test_transaction['card2'], color='blue', hist=False)



plt.show()
sns.distplot(np.log1p(train_transaction['card1']), color='red', hist=False)

sns.distplot(np.log1p(test_transaction['card2']), color='blue', hist=False)



plt.show()
plt.figure(figsize=(8,5))

p1 = sns.countplot(train_transaction['card3'],  order=train_transaction['card3'].value_counts().index[0:10])

for p in p1.patches:

        p1.annotate('{:6.2f}%'.format(p.get_height()/len(train_transaction)*100), (p.get_x()+0.1, p.get_height()+50))



plt.show()
p1 = sns.countplot(x= 'card4',data =train_transaction)

p1.set_xlabel('Card4 categorical names')

for p in p1.patches:

        p1.annotate('{:6.2f}%'.format(p.get_height()/len(train_transaction)*100), (p.get_x()+0.1, p.get_height()+50))

        

plt.show()
sns.countplot(x= 'card4',data =train_transaction[['card4','isFraud']], hue='isFraud')

plt.show()
plt.figure(figsize=(7,5))

plt.title('card4 Distribuition by ProductCD and Target')

sns.boxenplot(x='card4', y='TransactionAmt',data=train_transaction.loc[train_transaction['TransactionAmt']<2000], hue='isFraud')

plt.show()
fig, ax =plt.subplots(1,2 ,figsize=(10,5))

ax[0].set_title('Train')

ax[1].set_title('Test')

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)





p1 = sns.countplot(train_transaction['card4'].fillna('Missing'), ax=ax[0], palette ='plasma')

p2 = sns.countplot(test_transaction['card4'].fillna('Missing'), ax=ax[1], palette ='plasma')



for p in p1.patches:

        p1.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))

        



for p in p2.patches:

        p2.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))



fig.show()

plt.figure(figsize=(14,5))

p1 = sns.countplot(train_transaction['card5'],  order=train_transaction['card5'].value_counts().index[0:15], palette='plasma')

for p in p1.patches:

        p1.annotate('{:6.2f}%'.format(p.get_height()/len(train_transaction)*100), (p.get_x()+0.1, p.get_height()+50))



plt.show()
print("Card Features Quantiles:\n", train_transaction[['addr1','addr2']].quantile([.01, .025, 0.1, 0.25, 0.5, .75, 0.9, .975, .990]))

plt.text(500, 0.004, 'Train',  bbox=dict(facecolor='red', alpha=0.5))

plt.text(500, 0.0055, 'Test ', bbox=dict(facecolor='blue', alpha=0.5))



sns.distplot(train_transaction['addr1'], color='red', hist=False)

sns.distplot(test_transaction['addr1'], color='blue', hist=False)



plt.show()

%matplotlib inline

def count_boxplots(col):

    fig, ax =plt.subplots(1,2 ,figsize=(10,5))

    ax[1].set_title('Boxplot')

    ax[0].set_xticklabels(ax[0].get_xticklabels())

    ax[1].set_xticklabels(ax[1].get_xticklabels())





    p1 = sns.countplot(train_transaction[col].fillna('Missing'), ax=ax[0], palette ='plasma')

    p2 = sns.boxplot(x=col, y='TransactionAmt',data=train_transaction.loc[train_transaction[[col,'TransactionAmt','isFraud']].fillna('Missing')['TransactionAmt']<2000], hue='isFraud', ax=ax[1],\

                     palette ='plasma')



    for p in p1.patches:

        p1.annotate('{}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+50))

        

    fig.show()



for col in ['M1', 'M2',  'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:

    count_boxplots(col)
print(list(train_identity

           .columns))
train_transaction.head()
train_identity.head()
train_transaction_identity = train_transaction.merge(train_identity, on='TransactionID' )

test_transaction_identity = test_transaction.merge(test_identity, on='TransactionID' )
%matplotlib inline

def count_boxenplots(col):

    fig, ax =plt.subplots(3,1, figsize=(10,10) )

    ax[0].set_xticklabels(ax[0].get_xticklabels())

    ax[1].set_xticklabels(ax[1].get_xticklabels())

    ax[2].set_xticklabels(ax[2].get_xticklabels())



    p1 = sns.countplot(train_transaction_identity[col].fillna('Missing'), ax=ax[0], palette ='plasma')

    

    p2 = sns.countplot(x=col, data=train_transaction_identity[[col,'isFraud']].fillna('Missing'), hue='isFraud', ax=ax[1], palette ='plasma')



    p3 = sns.boxenplot(x=col, y='TransactionAmt',data= train_transaction_identity[[col,'TransactionAmt', 'isFraud']].fillna('Missing'), hue='isFraud', ax=ax[2],palette ='plasma')

        

    for p in p1.patches:

        p1.annotate('{:6.2f}%'.format(p.get_height()/len(train_transaction_identity)*100), (p.get_x()+0.3, p.get_height()+50))

        

    fig.show()



for col in ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29']:

    count_boxenplots(col)
fig, ax =plt.subplots(1,2 ,figsize=(10,5))

ax[0].set_title('Train')

ax[1].set_title('Test')

ax[0].set_xticklabels(ax[0].get_xticklabels())

ax[1].set_xticklabels(ax[1].get_xticklabels())





p1 = sns.countplot(train_identity['DeviceType'].fillna('Missing'), ax=ax[0], palette ='plasma')

p2 = sns.countplot(test_identity['DeviceType'].fillna('Missing'), ax=ax[1], palette ='plasma')



for p in p1.patches:

        p1.annotate('{}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+50))

        



for p in p2.patches:

        p2.annotate('{}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+50))



fig.show()
(np.sum(pd.isnull(train_transaction_identity)).sort_values(ascending=False)/len(train_transaction_identity))*100

#Rename column names in Test dataset to match with Train.

test_transaction_identity.rename(columns={ 'id-01':'id_01', 'id-02':'id_02', 'id-03':'id_03', 'id-04': 'id_04', \

                                          'id-05': 'id_05', 'id-06': 'id_06', 'id-07': 'id_07', \

                                          'id-08': 'id_08', 'id-09': 'id_09', 'id-10': 'id_10', 'id-11': 'id_11',\

                                          'id-12': 'id_12', 'id-13': 'id_13', 'id-14': 'id_14', 'id-15': 'id_15', 'id-16': 'id_16', 'id-17': 'id_17',\

                                          'id-18': 'id_18', 'id-19': 'id_19', 'id-20': 'id_20', 'id-21': 'id_21', 'id-22': 'id_22', 'id-23': 'id_23',\

                                          'id-24': 'id_24', 'id-25': 'id_25', 'id-26': 'id_26', 'id-27': 'id_27', 'id-28': 'id_28', 'id-29': 'id_29',\

                                          'id-30': 'id_30', 'id-31': 'id_31', 'id-32': 'id_32', 'id-33': 'id_33', 'id-34': 'id_34', 'id-35': 'id_35', \

                                          'id-36': 'id_36', 'id-37': 'id_37', 'id-38': 'id_38'}, inplace = True)
#Remove variables with missing values more than 30 percent



A = (np.sum(pd.isnull(train_transaction_identity)).sort_values(ascending=False)/len(train_transaction_identity))*100

Removed_col = A[A>0.3].index

train_transaction_identity.drop(columns=Removed_col, inplace=True)

test_transaction_identity.drop(columns=Removed_col,  inplace=True)
#Change Categorical Variables to dummies

train_dummy = pd.get_dummies(train_transaction_identity)
train_dummy.head()
#Remove redundant variables

train_dummy.drop(columns = 'TransactionID', inplace = True)
X_train, X_valid, y_train, y_valid = train_test_split(

    train_dummy.drop(columns='isFraud'), train_dummy['isFraud'], test_size=0.25, random_state=42)



params = {

    'objective' :'binary',

    'learning_rate' : 0.005,

    'num_leaves' : 192,

    'feature_fraction': 0.3, 

    'bagging_fraction': 0.7, 

    'bagging_freq':1,

    'boosting_type' : 'gbdt',

    'min_data_in_leaf': 100,

    'lambda_l1' : 0,

    'lambda_l2' : 0,

    'metric': 'auc'

}



d_train = lgb.Dataset(X_train, y_train)

d_valid = lgb.Dataset(X_valid, y_valid)

    

bst = lgb.train(params, d_train, 5000, valid_sets=[d_train, d_valid], verbose_eval=50, early_stopping_rounds=100)

    


