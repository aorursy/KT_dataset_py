# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="ticks")



import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()
missing_cols = train.isnull().sum()[train.isnull().sum() != 0].index.tolist()
train.drop(missing_cols, axis=1, inplace=True)

train.columns
cat_cols = [col for col in train.columns if train[col].dtype == object]
for i in np.arange(0, len(cat_cols)-1, 2):

    fig, axes = plt.subplots(1, 2, figsize = (18, 6))

    plt.subplot(1, 2, 1);    

    sns.scatterplot(x=cat_cols[i], y="SalePrice", data=train)

    plt.xticks(rotation=30)



    plt.subplot(1,2,2)

    sns.scatterplot(x=cat_cols[i+1], y="SalePrice", data=train)

    plt.xticks(rotation=30)  

    plt.ylabel('')

num_cols = [col for col in train.columns if train[col].dtype == 'int64']
num_cols.remove('Id')

num_cols.remove('SalePrice')
for i in np.arange(0, len(num_cols)-1, 5):

    g = sns.pairplot(train[num_cols[i:i+5]+['SalePrice']], palette="Set2", diag_kind="kde", height=2.5) 