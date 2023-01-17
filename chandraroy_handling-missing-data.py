# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Any results you write to the current directory are saved as output.
# Reading the dataset
df_original = pd.read_csv("../input/Melbourne_housing_FULL.csv")
df_original.head()
df_original.isnull().sum()
df_original.describe()
df_original.dtypes
df_sub = df_original.drop('Regionname', 1)
df_sub = df_sub.drop('CouncilArea', 1)
df_sub.head(2)
df_sub.shape
from sklearn.preprocessing import Imputer

temp_imputer = Imputer()
Price_imputed = temp_imputer.fit_transform(df_sub[['Price']])
df_sub[['Price']] = Price_imputed
df_sub[['Price']].isnull().sum()
df_sub.head(100)
df_original.head(100)
import matplotlib.pyplot as plt

plt.figure(figsize = (20, 20))

plt.hist(df_sub[['Price']], bins = 10, edgecolor = 'black', linewidth = 2)
plt.title("Distribution of House Price")
plt.xlabel('Number of Units')
plt.ylabel('Price')
plt.show()