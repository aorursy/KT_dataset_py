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
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 



import warnings

# current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')



import missingno as msno # missing data visualization module for Python

import pandas_profiling



import gc

import datetime



%matplotlib inline

color = sns.color_palette()

pd.set_option('display.max_rows', 10000)

pd.set_option('display.max_columns', 100)

# specify encoding to deal with different formats

train=pd.read_csv('/kaggle/input/fmcgfast-moving-consumer-goods/train1.csv',index_col=0)

test=pd.read_csv('/kaggle/input/fmcgfast-moving-consumer-goods/test1.csv',index_col=0)

test.head()
#########Data Cleaning

train.info()

test.info()
#########Check missing values for each column

train.isnull().sum().sort_values(ascending=False)

test.isnull().sum().sort_values(ascending=False)



# check out the rows with missing values

train[train.isnull().any(axis=1)].head()
# check out the rows with missing values

test[test.isnull().any(axis=1)].head()

#######Remove rows with missing values

train_new = train.dropna()



test_new = test.dropna()
# check missing values for each column 

train_new.isnull().sum().sort_values(ascending=False)

## check missing values for each column 

test_new.isnull().sum().sort_values(ascending=False)

train_new.info()


test_new.info()

train_new.describe()

test_new.describe().round(2)

#how many unique kinds of products are there?

print("There are {} unique products are there in dataset.".format(len(train_new.PROD_CD.unique())))
#how many unique kinds of products are there?

print("There are {} unique products are there in dataset.".format(len(test_new.PROD_CD.unique())))
#how many salespersons are there?

print("There are {} salesperson.".format(len(train_new.SLSMAN_CD.unique())))
#how many salespersons are there?

print("There are {} salesperson.".format(len(test_new.SLSMAN_CD.unique())))
train_new.boxplot(column=['PLAN_YEAR'])

train_new.PLAN_MONTH.plot.hist()

train_new.PLAN_YEAR.plot.hist()

plt.figure(figsize=(10,7))

chains=train_new['SLSMAN_CD'].value_counts()[:20]

sns.barplot(x=chains,y=chains.index,palette='deep')



plt.title("top 20 salesperson")

plt.xlabel("No. of apperence in data")

test_new.boxplot(column=['PLAN_YEAR'])

test_new.PLAN_MONTH.plot.hist()

test_new.PLAN_YEAR.plot.hist()

plt.figure(figsize=(10,7))

chains=test_new['SLSMAN_CD'].value_counts()[:20]

sns.barplot(x=chains,y=chains.index,palette='deep')



plt.title("top 20 salesperson")

plt.xlabel("No. of apperence in data")

####Check TOP 5 most number of target achive

print('The TOP 5 salesman with most number of target given to the salepearson...')

train_new.sort_values(by='TARGET_IN_EA', ascending=False).head()

####Check TOP 5 most number target in eachive

print('The TOP 5 salesman with most number of target given to the salepearson...')

test_new.sort_values(by='TARGET_IN_EA', ascending=False).head()

####Check TOP 5 most number of sold product in perticular month

print('The TOP 5 salesman with most number of sold product in perticular month given to the salepearson...')

train_new.sort_values(by='ACH_IN_EA', ascending=False).head()
