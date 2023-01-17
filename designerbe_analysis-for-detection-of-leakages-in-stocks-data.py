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
#importing the data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
data_cleaner = [df_train,df_test]
df_train.head().transpose()
df_train.describe()
df_train['time'].describe() #the date is in object format. Let us convert to datetime format for analysis
#It can be done directly as the data has been provided in a convertible format

df_train['time']= pd.to_datetime(df_train['time']) 

df_test['time']= pd.to_datetime(df_test['time']) 
df_train['time'].describe()
df_test['time'].describe()
df_train.shape[0]
#Let us now explore the price dataset.

for dataset in data_cleaner:

    dataset['pricebin'] = pd.cut(dataset['price'],df_train.shape[0]/2500)
df_train['pricebin'].unique()
pricebin_tr = df_train['pricebin'].value_counts(sort=False)

pricebin_t = df_test['pricebin'].value_counts(sort=False)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].set_ylabel('Frequency')

axes[0].set_title('Training set')

plt.ylabel('Frequency')

pricebin_tr.plot(ax=axes[0],kind='bar')

axes[1].set_ylabel('Frequency')

axes[1].set_title('Testing set')

pricebin_t.plot(ax=axes[1],kind='bar')

fig.tight_layout()
df_test['pricebin'].unique()
df_test.describe()
fig, ax = plt.subplots()

sns.kdeplot(df_train['price'], ax=ax,shade=True)

sns.kdeplot(df_test['price'], ax=ax,shade=True)
for dataset in data_cleaner:

    #skewness and kurtosis

    print("Skewness: %f" % dataset['price'].skew())

    print("Kurtosis: %f" % dataset['price'].kurt())