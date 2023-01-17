# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/BlackFriday.csv')
df.head()
df.columns
number_of_customers = df.User_ID.unique().size

print('Total number of customers: ',number_of_customers)
group_by_gender = df.loc[:,['User_ID','Gender']].drop_duplicates()

(group_by_gender['Gender'].value_counts()/len(group_by_gender['Gender'])).plot.bar()

df.Purchase.sum()
(df.loc[:,['Gender','Purchase']].groupby('Gender').sum()/df.Purchase.sum()).plot.bar()
df.City_Category.unique()
(df.loc[:,['City_Category','Purchase']].groupby('City_Category').sum()/df.Purchase.sum()).plot.bar()
age_groups = df.Age.unique()

print('Different age groups: ',age_groups)
purchase_age = df[['Age','Purchase']]

print(purchase_age.groupby('Age').sum())

(purchase_age.groupby('Age').sum()/1000000).plot.bar(title ='Purchase by age in million dollars')
product_categories = df[['Product_Category_1','Product_Category_2','Product_Category_3']]

product_categories.head()
sum_categories = product_categories.sum(axis =0,skipna = True)

print(sum_categories)

(sum_categories/sum_categories.sum()).plot.bar(title = 'Percent of sale by category')
df.Stay_In_Current_City_Years.unique()
(df[['Stay_In_Current_City_Years','Purchase']].groupby('Stay_In_Current_City_Years').sum()/1000000).plot.bar()
(df[['Marital_Status','Purchase']].groupby('Marital_Status').sum()/1000000).plot.bar()
sns.countplot(df['City_Category'],hue= df['Age'])
sns.countplot(df['Occupation'])