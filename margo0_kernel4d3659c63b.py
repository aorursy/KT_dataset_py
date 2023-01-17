# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv",index_col=0)

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv",index_col=0)
train.head()
test.head()
train.describe()
test.describe()
train.shape
test.shape
sample_submission.shape
train.isna().sum()
test.isna().sum()
train.columns
test.columns
plt.scatter(train.iloc[:50,1:14],test.iloc[:50,:13],color='blue')
sns.countplot(train['target'])
plt.figure(figsize=(16,6))

features = train.columns.values[2:202]

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train[features].mean(axis=1),color="red", kde=True, label='train')

sns.distplot(test[features].mean(axis=1),color="blue", kde=True, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="purple",kde=True,bins=100, label='train')

sns.distplot(test[features].mean(axis=0),color="darkgreen", kde=True,bins=100, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=1),color="black", kde=True,bins=100, label='train')

sns.distplot(test[features].std(axis=1),color="orange", kde=True,bins=100, label='test')

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per column in the train and test set")

sns.distplot(train[features].std(axis=0),color="green",kde=True,bins=100, label='train')

sns.distplot(test[features].std(axis=0),color="yellow", kde=True,bins=100, label='test')

plt.legend(); plt.show()
t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per row in the train set")

sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=100, label='target = 0')

sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=100, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train set")

sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=0),color="brown", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()



plt.figure(figsize=(16,6))

features = train.columns.values[2:202]

plt.title("Distribution of min values per row in the train and test set")

sns.distplot(train[features].min(axis=1),color="red", kde=True,bins=100, label='train')

sns.distplot(test[features].min(axis=1),color="yellow", kde=True,bins=100, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train.columns.values[2:202]

plt.title("Distribution of min values per column in the train and test set")

sns.distplot(train[features].min(axis=0),color="magenta", kde=True,bins=100, label='train')

sns.distplot(test[features].min(axis=0),color="black", kde=True,bins=100, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train.columns.values[2:202]

plt.title("Distribution of max values per row in the train and test set")

sns.distplot(train[features].max(axis=1),color="blue", kde=True,bins=100, label='train')

sns.distplot(test[features].max(axis=1),color="yellow", kde=True,bins=100, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train.columns.values[2:202]

plt.title("Distribution of max values per column in the train and test set")

sns.distplot(train[features].max(axis=0),color="blue", kde=True,bins=120, label='train')

sns.distplot(test[features].max(axis=0),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()
t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of min values per row in the train set")

sns.distplot(t0[features].min(axis=1),color="orange", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].min(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of min values per column in the train set")

sns.distplot(t0[features].min(axis=0),color="red", kde=True,bins=100, label='target = 0')

sns.distplot(t1[features].min(axis=0),color="blue", kde=True,bins=100, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of max values per row in the train set")

sns.distplot(t0[features].max(axis=1),color="gold", kde=True,bins=100, label='target = 0')

sns.distplot(t1[features].max(axis=1),color="blue", kde=True,bins=100, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of max values per column in the train set")

sns.distplot(t0[features].max(axis=0),color="red", kde=True,bins=100, label='target = 0')

sns.distplot(t1[features].max(axis=0),color="blue", kde=True,bins=100, label='target = 1')

plt.legend(); plt.show()