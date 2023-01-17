# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Imporing the required Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Import Train and Test files



train=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head(10)
train.info()
#Ploting the missing values using Heatmap



plt.figure(figsize = (10,6))

sns.heatmap(train.isnull())
# To find the number of Missing Values in each Columns



train.isnull().sum()
#Dropping columns which have more than 50% Missing Values



train=train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)  

test = test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
train.isnull().sum()
# Separating Numerical and Categorical columns



train_num_cols = train.select_dtypes(exclude='object').columns

train_cat_cols = train.select_dtypes(include='object').columns





test_num_cols = test.select_dtypes(exclude='object').columns

test_cat_cols = test.select_dtypes(include='object').columns
train_num_cols
train_cat_cols
#Filling Missing Values (Numerical Features) using mean of all non null values.



for i in range(0, len(train_num_cols)):

    train[train_num_cols[i]] = train[train_num_cols[i]].fillna(train[train_num_cols[i]].mean())

    

for i in range(0, len(test_num_cols)):

    test[test_num_cols[i]] = test[test_num_cols[i]].fillna(test[test_num_cols[i]].mean())
#Filling Missing Values (Categorical Features) using mode of all non null values.



for i in range(0, len(train_cat_cols)):

    train[train_cat_cols[i]] = train[train_cat_cols[i]].fillna(train[train_cat_cols[i]].mode()[0])

    

for i in range(0, len(test_cat_cols)):

    test[test_cat_cols[i]] = test[test_cat_cols[i]].fillna(test[test_cat_cols[i]].mode()[0])