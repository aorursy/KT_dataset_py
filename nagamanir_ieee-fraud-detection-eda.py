# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_identity = pd.read_csv('../input/train_identity.csv')

train_transaction = pd.read_csv('../input/train_transaction.csv')
## Function to reduce the DF size. Re-used from  https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt/notebook

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_identity = reduce_mem_usage(train_identity)

train_transaction = reduce_mem_usage(train_transaction)
train_transaction.head()
total = len(train_transaction)

ax = sns.countplot(x='isFraud', data=train_transaction)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center") 
f, axes = plt.subplots(1, 2, figsize=(18, 6))

card4_ax = sns.countplot(x='card4', hue='isFraud', data=train_transaction, ax=axes[0])

card6_ax = sns.countplot(x='card6', hue='isFraud', data=train_transaction, ax=axes[1])

for ax in axes:

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center") 
# cards 1,2,3,5 

f, axes = plt.subplots(4, 1, figsize=(18, 8))

card1_ax = sns.distplot(train_transaction['card1'], ax=axes[0])

card2_ax = sns.distplot(train_transaction['card2'].dropna(), ax=axes[1])

card3_ax = sns.distplot(train_transaction['card3'], ax=axes[2], kde=False)

card4_ax = sns.distplot(train_transaction['card5'].dropna(), ax=axes[3])
f, axes = plt.subplots(1, 2, figsize=(18, 6))

addr1 = sns.distplot(train_transaction['addr1'].dropna(), ax=axes[0], kde=False)

addr2 = sns.distplot(train_transaction['addr2'], ax=axes[1], kde=True)
g = sns.FacetGrid(train_transaction, col="isFraud")

g.map(sns.distplot, "addr1")
g = sns.FacetGrid(train_transaction, col="isFraud")

g.map(sns.distplot, "addr2")
#correlation between addr1 and addr2

# copied the code from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

from scipy import stats as ss

def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



cramers_v(train_transaction['addr1'], train_transaction['addr2'])
ax = sns.countplot(x='ProductCD', hue='isFraud', data=train_transaction )

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=10) 
sns.boxenplot(x="ProductCD", y='TransactionAmt', hue='isFraud',data=train_transaction.sample(2000))
f, axes = plt.subplots(1, 2, figsize=(18, 12))



sns.set(color_codes=True)

p_email = sns.countplot(y='P_emaildomain', data=train_transaction, ax=axes[0])

r_email = sns.countplot(y='R_emaildomain', data=train_transaction, ax=axes[1])
p_df = train_transaction['P_emaildomain'].value_counts()

p_df = p_df.reset_index()

p_df.columns = ['emaildomain', 'count']

p_df = p_df[p_df['count'] > 2000]



r_df = train_transaction['R_emaildomain'].value_counts()

r_df = r_df.reset_index()

r_df.columns = ['emaildomain', 'count']

r_df = r_df[r_df['count'] > 800]
f, axes = plt.subplots(1, 2, figsize=(18, 12))



#sns.set(color_codes=True)

p_email = sns.countplot(y='P_emaildomain', data=train_transaction[train_transaction['P_emaildomain'].isin(p_df['emaildomain'].values)], hue = 'isFraud', ax=axes[0])

r_email = sns.countplot(y='R_emaildomain', data=train_transaction[train_transaction['R_emaildomain'].isin(r_df['emaildomain'].values)], hue = 'isFraud', ax=axes[1])
col_ls =  ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

f, axes = plt.subplots(3, 3, figsize=(18, 12))

count = 0

for i in range(3): # rows loop

    for j in range(3): # cols loop

        mplot = sns.countplot(x=col_ls[count], hue = 'isFraud', data=train_transaction, ax=axes[i,j])

        count += 1 # to loop over col-names

        

for ax_r in axes:

    for ax in ax_r:

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center") 
g = sns.FacetGrid(train_transaction, col="isFraud")

g.map(sns.kdeplot, "TransactionAmt")
# since the distribution is very skwed, lets check for the amounts greate than 10000



tr_amt = train_transaction[train_transaction['TransactionAmt'] > 10000]

tr_amt
sns.distplot(train_transaction[train_transaction['TransactionAmt'] <= 6000]['TransactionAmt'])
#lets check for the distribution for the transactin amoutn < 1000 as most of the transactions have amount less than 100

g = sns.FacetGrid(train_transaction[train_transaction['TransactionAmt'] <= 1000], col="isFraud")

g.map(sns.kdeplot, "TransactionAmt", )