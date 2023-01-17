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



# Any results you write to the current directory are saved as output

train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# back up

train = train_df.copy()

test = test_df.copy()
train_df
# Get catagorical columns for object data (For example: Strings)

print(train_df.describe(include='all').loc['unique',:].dropna())

# Get catagorical columns for numeric data

numeric_features_list = ["Id", "SalePrice", "LotArea"]   

train_df.drop(numeric_features_list, 1).select_dtypes(np.number).nunique()

train_df.columns[train_df.isna().any()].tolist()
colnames_numerics_only = train_df.select_dtypes(include=np.number).columns.tolist()

print(colnames_numerics_only)

# Substract Catagorical Features from _ 

a = train_df.corr('pearson')

a[a.SalePrice>0.5]