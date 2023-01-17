# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import the dataset as a pandas dataframe
data = pd.read_csv("../input/BlackFriday.csv")
data.head(5)
data.info()
# Check for null values (e.g. NaN)
data.isnull().any()
data = data.fillna(0)
# Now we set the features to the right datatypes
data['User_ID'] = data['User_ID'].astype(str)
data['Occupation'] = data['Occupation'].astype(str)
data['Marital_Status'] = data['Marital_Status'].astype(str)
data['Product_Category_2'] = data['Product_Category_2'].astype(int)
data['Product_Category_3'] = data['Product_Category_3'].astype(int)
# General statistical information about the dataset
data.describe(include=['object'])
data.describe()
count_data = data[['Gender','Purchase']].groupby('Gender').count()
count_data = pd.DataFrame(count_data.to_records())   # VERY USEFUL TRICK
count_data.columns = ['Gender','Purchase_count']
count_data.head()
sns.barplot(count_data.Gender,count_data.Purchase_count)
purchase_count = data[['Gender','Purchase']].groupby('Gender').sum()
purchase_count = pd.DataFrame(purchase_count.to_records())   # VERY USEFUL TRICK
purchase_count
# Lets show it in percentage
total_sum = data['Purchase'].sum()
purchase_count.columns = ['Gender','Purchase_sum']
sns.barplot(purchase_count.Gender,purchase_count.Purchase_sum*100/total_sum)
sns.catplot(x='Gender', y = 'Purchase', data = data, kind='violin')
fig, axes = plt.subplots(nrows = 2, ncols = 2)
sns.countplot(data['Age'], ax = axes[0,0])
sns.countplot(data['City_Category'], ax = axes[0,1])
sns.countplot(data['Occupation'], ax = axes[1,0])
sns.countplot(data['Marital_Status'], ax = axes[1,1])
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.75,
                    wspace=0.75)
sns.distplot(data['Purchase'],bins=25,kde=False)
data.groupby('Product_ID')['Product_ID'].count().sort_values(ascending=False).head(5)
data.groupby('User_ID')['User_ID'].count().sort_values(ascending=False).head(5)
data[['User_ID','Purchase']].groupby('User_ID').sum().sort_values(ascending=False,by=['Purchase']).head(5)
