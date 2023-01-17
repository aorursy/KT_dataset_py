# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Read the dataset to pandas DataFrame
df = pd.read_excel("../input/Online Retail.xlsx")

#Explore the data set
df.head()
df1 = df
#Transform columns to lower case

df1.columns = df1.columns.str.lower()
df1.columns
print("Total data points: ", df1.shape[0])
print("Total missing values: {} - which is {:.2f}% of our total data".format(df1.isnull().sum().sum(), (df1.isnull().sum().sum()*100)/df1.shape[0]))
print("Total unique Countries: ", df1.country.nunique())
print("Total unique description: ", df1.description.nunique())
#Top 5 Most common countries

df1.country.value_counts()[:5].plot(kind='bar')
#Top 5 least common countries

df1.country.value_counts()[-5:].plot(kind='bar')
# Top 10 product description

df1.description.value_counts()[:10]
#Count of missiong vaues

df1.isnull().sum()
# Drop all missing values

df1.dropna(inplace=True)
df1.isnull().sum()
print("Total numbers of Features: {} and Data points: {}".format(df1.shape[1], df1.shape[0]))
df1.head()
df1.head()
#df1['customerid'].astype(int, inplace=True)
#df1['customerid'].dtypes
# Gropping countries by Total quantity

df1.groupby('country')['quantity'].sum().sort_values(ascending=False)
# Top 5 countries by Total quantity

df1.groupby('country')['quantity'].sum().sort_values(ascending=False)[:5].plot(kind='bar')
# Top 5 countries by Total unitprice

df1.groupby('country')['unitprice'].sum().sort_values(ascending=False)[:5].plot(kind='bar')
# Create a new Features 'year' from 'invoicedate'

df1['year'] = df1['invoicedate'].dt.year
df1.head()
# Total unitprice sold by year

df1.groupby('year')['unitprice'].sum().plot(kind='bar')
# Total quantity sold by year

df1.groupby('year')['quantity'].sum().plot(kind='bar')
#Total quantity sold by invoiceno [Top 10]

df1.groupby('invoiceno')['quantity'].sum().sort_values(ascending=False)[:10].plot(kind='bar')



