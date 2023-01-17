# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as pl

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_identity = pd.read_csv("../input/train_identity.csv")

train_transaction = pd.read_csv("../input/train_transaction.csv")

test_identity = pd.read_csv("../input/test_identity.csv")

test_transaction = pd.read_csv("../input/test_transaction.csv")
train_transaction.head()
print(train_transaction.info())
train_transaction.isnull().sum()
#Let's merge our data sets for Future!

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
clean_df = train_transaction.dropna(axis=1)

clean_df.head()
print(clean_df.shape)

print(train_transaction.shape)
clean_df.isnull().sum()
# Wow, the data looks good to go!

# let's check "isFraud" data balance in respect to our target

target_balancing = clean_df['isFraud'].value_counts().values

sns.barplot([0,1],target_balancing);

pl.title("How much is our target balanced??");
#Let's try seeing the Fraud and non fraud transactions based on Products

sns.barplot(x = clean_df.index,y = 'ProductCD',hue='isFraud',data = clean_df);
pl.figure(figsize=(16,7))

sns.boxplot(data = clean_df.drop(columns = ['TransactionID','isFraud']))

pl.yscale('log')

pl.xticks(rotation=90);
pl.hist(train['TransactionDT'], label='train');

pl.hist(test['TransactionDT'], label='test');

pl.legend();

pl.title('Distribution of transaction dates');
train.head()
clean_df.head()
pl.figure(figsize=(16,7))

cor  =clean_df.corr()

sns.heatmap(cor,annot=True);
fraudlent = clean_df[clean_df['isFraud']==1]

non_fraudlent = clean_df[clean_df['isFraud']==0]

fraudlent.head()
#Let's see the Transaction AMount for the Fraudlent and Non Fraudlent Transactions!

pl.figure(figsize=(16,7))

pl.subplot(1,2,1)

sns.distplot(fraudlent['TransactionAmt'],rug=True);

pl.title('Transaction Amount for Fraudlent Transactions!');

pl.subplot(1,2,2)

sns.distplot(non_fraudlent['TransactionAmt'],rug=True);

pl.title('Transaction Amount for Non Fraudlent Transactions!');
mean_fraud = fraudlent['TransactionAmt'].mean()

mean_non_fraud = non_fraudlent['TransactionAmt'].mean()

print(mean_fraud)

print(mean_non_fraud)
# Correlation for each

cor_fraud = fraudlent.corr()

cor_non_fraud = non_fraudlent.corr()

pl.figure(figsize=(20,10))

pl.subplot(2,1,1)

sns.heatmap(cor_fraud,annot=True)

pl.subplot(2,1,2)

sns.heatmap(cor_non_fraud,annot=True);
# We will go on to find the Top 5 Attributes that are highly Correlated fro each of the Category!

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations for Fraudlent Transactions!")

print(get_top_abs_correlations(fraudlent.drop(columns = ['ProductCD','isFraud']), 5))



# I love this peice of code as it makes the Corelation much easier to look!
# We will go on to find the Top 5 Attributes that are highly Correlated fro each of the Category!

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations for Non Fraudlent Transactions!")

print(get_top_abs_correlations(non_fraudlent.drop(columns = ['ProductCD','isFraud']), 5))
train_identity.head()
# Let's get rid of Null values on AXIS = 1

train_identity_clean_df  = train_identity.dropna(axis=0).reset_index(drop=True)

train_identity_clean_df.head()
train_identity_clean_df.isnull().sum()
# Well I'm thinking of mergign the cleaned Transaction and Identity dataframes as this gives us better view

merged_train = pd.merge(clean_df,train_identity_clean_df,on = 'TransactionID',how='left')

merged_train.head()
merged_train = merged_train.dropna(axis = 0).reset_index(drop=True)

merged_train.head()
merged_train.shape
balancing_merge_df = merged_train['isFraud'].value_counts().values

sns.barplot([0,1],balancing_merge_df);

pl.title('Checking balance of our target in the merged Data Set!');
merged_train['TransactionAmt'].describe()
pl.figure(figsize=(16,7))

pl.subplot(1,2,1)

pl.plot('TransactionAmt','ProductCD', data=merged_train, marker='o', alpha=0.4)

pl.subplot(1,2,2)

pl.plot('TransactionAmt','ProductCD', data=merged_train, linestyle='none', marker='o', color="orange", alpha=0.3);

pl.title('Different types of Products on Transaction Amount!');
sns.set_style("white")

sns.kdeplot(merged_train.TransactionAmt, merged_train.card1);
# top correlating Attributes for the merged Data Frame!

pl.figure(figsize=(20,10))

corr_merge = merged_train.corr()

sns.heatmap(corr_merge,annot=True)

print("Top Absolute Correlations for Merged Data frame!!")

numeric_df_merged = merged_train.select_dtypes(include=['int'])

print(get_top_abs_correlations(numeric_df_merged, 5))