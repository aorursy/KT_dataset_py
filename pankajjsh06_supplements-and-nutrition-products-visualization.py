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
df = pd.read_csv('../input/bodybuilding_nutrition_products.csv')
df.info()
df.isnull().any()
df.fillna(0,inplace=True)
df.head()
df.drop(['link','product_description','verified_buyer_number'], axis=1,inplace = True)
df.head()
print(df['brand_name'].unique())

print(df['product_category'].unique())

print(df['product_name'].unique())

print(df['top_flavor_rated'].unique())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
corr = df.corr()

plt.figure(figsize=(12,6))

sns.heatmap(corr,annot=True)

plt.show()
plt.figure(figsize=(12,6))

sns.pairplot(df)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot('product_category',data=df,palette='viridis', order = df['product_category'].value_counts().index)

plt.xticks(rotation=90,ha='right')

plt.tight_layout()

plt.show()
df.columns
df['product_category'].replace([0],['unnamed'], inplace=True);


brand_df = df[['average_flavor_rating', 'brand_name', 'number_of_flavors','product_category']]

brand_df.head()