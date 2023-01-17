# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
color=['r', 'g', 'b', 'k', 'm', 'y']
sales = pd.read_csv('../input/BlackFriday.csv')
sales.head()
sales.info()
sales.describe()
sales.hist(figsize=(20,15))
sales.columns
sales['Gender'].value_counts().plot.bar(color=color)
sales['Marital_Status'].value_counts().plot.bar(color=color)
sales['Age'].value_counts().plot.hist()
sales['Age'].unique()
sales.columns
(sales['Product_Category_1'].unique())
gender_marital_status_grouped = sales.groupby(['Gender','Marital_Status']).size().reset_index(name="Total")
gender_marital_status_grouped.head()
gender_marital_status_grouped.set_index(['Gender','Marital_Status']).unstack().plot(kind='bar', stacked=True)
gender_marital_status_grouped.set_index(['Marital_Status','Gender']).unstack().plot(kind='bar', stacked=True, colormap='Paired')
gender_marital_status_grouped.set_index(['Marital_Status','Gender']).unstack().plot(kind='bar')
sales.columns
sales['Stay_In_Current_City_Years'].unique()
total_purchase = sales.groupby(['Gender','Marital_Status']).sum()['Purchase'].reset_index(name="Total")
total_purchase.head()
total_purchase.set_index(['Gender','Marital_Status']).unstack().plot(kind='bar', stacked=True)
total_purchase.set_index(['Marital_Status','Gender']).unstack().plot(kind='bar', stacked=True)
purchase_category_1_marital_status = sales.groupby(['Product_Category_1','Marital_Status',]).size().reset_index(name="Total")
purchase_category_1_marital_status.head(5)
purchase_category_1_marital_status.set_index(['Product_Category_1', 'Marital_Status']).unstack().plot(kind='bar', stacked=True)
purchase_category_1_gender = sales.groupby(['Product_Category_1','Gender',]).size().reset_index(name="Total")
purchase_category_1_gender.set_index(['Product_Category_1', 'Gender']).unstack().plot(kind='bar', stacked=True)
sales['Product_Category_1'].unique()
(sales[['Product_Category_2','Product_Category_3']].isnull().sum()/sales.shape[0]*100).reset_index(name="Total").plot.bar(x='index', y='Total', color=['r','b'], title='% of Null Values')
purchase_category_1 = (sales.groupby(['Product_Category_1']).sum()['Purchase']/1000000).reset_index(name="Total")
purchase_category_1.plot.bar(x='Product_Category_1', y='Total', title='Purchase Value for Cateogry 1')
purchase_category_2 = (sales.groupby(['Product_Category_2']).sum()['Purchase']/1000000).reset_index(name="Total")
purchase_category_2.plot.bar(x='Product_Category_2', y='Total', title='Purchase Value for Cateogry 2')
purchase_category_3 = (sales.groupby(['Product_Category_3']).sum()['Purchase']/1000000).reset_index(name="Total")
purchase_category_3.plot.bar(x='Product_Category_3', y='Total', title='Purchase Value for Cateogry 3')
purchase_by_age = (sales.groupby(['Age']).sum()['Purchase']/1000000).reset_index(name="Total")
purchase_by_age.plot.bar(x='Age', y='Total', title='Purchase Value by age')
age_category1 = (sales.groupby(['Age','Product_Category_1']).size()).reset_index(name="Total")
age_category1.set_index(['Product_Category_1','Age']).unstack().plot(kind='bar', stacked=True, figsize=(20,15))
age_category1 = (sales.groupby(['Age','Product_Category_2']).size()).reset_index(name="Total")
age_category1.set_index(['Product_Category_2','Age']).unstack().plot(kind='bar', stacked=True, figsize=(20,15))
age_category1 = (sales.groupby(['Age','Product_Category_3']).size()).reset_index(name="Total")
age_category1.set_index(['Product_Category_3','Age']).unstack().plot(kind='bar', stacked=True, figsize=(20,15))
