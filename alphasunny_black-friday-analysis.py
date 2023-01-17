# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpt

import matplotlib.pyplot as plt

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/BlackFriday.csv")
df.info()
df.head()
df.isna().any()
print("Product_Category_1", df['Product_Category_2'].unique())

print("Product_Category_1", df['Product_Category_3'].unique())

df.fillna(0,inplace=True)
df.isna().any()
sb.countplot(df['Gender'])
sb.countplot(df['Age'], hue=df['Gender'])
df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)

print(df['combined_G_M'].unique())
sb.countplot(df['Age'],hue=df['combined_G_M'])
df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)

print(df['combined_G_M'].unique())
sb.countplot(df['Age'],hue=df['combined_G_M'])
sb.countplot(df['Product_Category_2'],hue=df['combined_G_M'])