# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Number of rows and columns in train set : ",train.shape)

print("Number of rows and columns in test set : ",test.shape)
pd.set_option('max_columns',258)

train.head()
test.head()
sns.countplot(train['target'], palette='Set3')
train.target.value_counts()
pd.set_option('max_rows',258)

train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
feats = [f for f in train.columns if f not in ['id','target']]

for i in feats:

    print ('==' + str(i) + '==')

    print ('train:' + str(train[i].nunique()/train.shape[0]))

    print ('test:' + str(test[i].nunique()/test.shape[0]))
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(26,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(26,10,i)

        sns.distplot(df1[feature], hist=False,label=label1)

        sns.distplot(df2[feature], hist=False,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();

    

t0 = train[feats].loc[train['target'] == 0]

t1 = train[feats].loc[train['target'] == 1]

features = train[feats].columns.values

plot_feature_distribution(t0, t1, '0', '1', features)    
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="yellow",kde=True,bins=50, label='train')

sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=50, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=0),color="red", kde=True,bins=50, label='train')

sns.distplot(test[features].std(axis=0),color="yellow", kde=True,bins=50, label='test')

plt.legend()

plt.show()
correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.head(10)
correlations.tail(10)
feats_target = [f for f in train.columns if f not in ['id']]

correlations = train[feats_target].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

corr = correlations[correlations['level_0']=='target']

corr.head(300)