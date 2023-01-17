# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/googleplaystore.csv', header=0)
df.head()
df.dtypes
df['Price'].value_counts()
df['SizeS'] = df['Size'].str.extract('([0-9\.]+)[Mk]')
df['PriceS'] = df['Price']
df['PriceS'] = np.where(df['Price'] != '0', df['Price'].str.extract('\$([0-9\.]*)'), df['Price'])
df.head()
df['Ratingf'] = df['Rating'].astype("float64")
df['Pricef'] = df['PriceS'].astype("float64")
df['Reviewsf'] = df['Reviews'].astype("float64", errors="ignore")
df['Sizef'] = df['SizeS'].astype("float64")
df['Last Updatedf'] = pd.to_datetime(df['Last Updated'], format='%B %d, %Y', errors="ignore")
df.head()
df.dtypes
print(df.shape)
df.isna().sum()
df['Category'].value_counts()
df_fill = df
df_fill['Sizem'] = df_fill['Sizef'].fillna(df_fill.groupby("Category")['Sizef'].transform('mean'))
df_fill['Ratingm'] = df_fill['Ratingf'].fillna(df_fill.groupby("Category")['Ratingf'].transform('mean'))
df_fill['Pricem'] = df_fill['Pricef'].fillna(df_fill.groupby("Category")['Pricef'].transform('mean'))
df_fill.head()
df_fill = df_fill.dropna(how='any', subset=['Ratingm', 'Sizem', 'Pricem'])
df_fill.isna().sum()
df_fill.head()