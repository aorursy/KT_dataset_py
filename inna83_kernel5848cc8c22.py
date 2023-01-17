import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

print("Setup Complete")

import os

os.listdir('../input')

mall_customers = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

mall_customers.head()

#data types check 

mall_customers.dtypes
mall_customers.tail()
#Null control

print(mall_customers.isnull().sum())
mall_customers.head()
mall_customers.describe()
plt.figure(figsize=(10,6))

plt.title('Gender vs Spending Score')

sns.barplot(x=mall_customers['Gender'],y=mall_customers['Spending Score (1-100)'])

plt.figure(figsize=(10,6))

plt.title('Age vs Spending Score')

sns.scatterplot(x= mall_customers['Age'], y= mall_customers['Spending Score (1-100)'])



plt.figure(figsize=(10,6))

plt.title('Age vs Spending Score')

sns.regplot(x= mall_customers['Age'], y= mall_customers['Spending Score (1-100)'])
#Conclusion: Based on analyse - women spend money faster than men. The older people are, the slower they spend money. People by 40 years spend money faster.