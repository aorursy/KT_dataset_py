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



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

df=pd.read_excel('/kaggle/input/US Superstore data.xls')

df.head()
#EDA

#row col count

df.shape
#col names

df.columns
#dtypes of cos

df.dtypes
#if there are missing valuse

df.isnull().sum()
#drop row id col

df=df.drop('Row ID', axis=1)

df.head()
df['Country'].value_counts()
#droppig country col beacuse data is for us only

df=df.drop('Country', axis=1)

df.head()
#product categories

df['Category'].unique()
#no. of products in each category

df['Category'].value_counts()
df['Sub-Category'].nunique()
df['Sub-Category'].value_counts()
#let see how sub category wrt ategory is seen in graph

plt.figure(figsize=(16,8))

plt.bar('Sub-Category', 'Category', data=df, color='g')

plt.show()
#lets see it in pie chart

plt.figure(figsize=(12,10))

df['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
df.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar()

plt.show()
#Highest profit is in Copiers 

#Sales is in Chairs and Phones 
#no of products in store

df['Product Name'].value_counts()
df['Product Name'].nunique()
plt.figure(figsize=(12,10))

df['Product Name'].value_counts().head(10).plot.pie(autopct='%1.1f%%')

#count of sub category region wise

plt.figure(figsize=(15,8))

sns.countplot(x='Sub-Category', hue='Region', data=df)

plt.show()
#CP=SP-Profit

#Creating CP nw col

df['CP']=df['Sales']-df['Profit']

df['CP'].head()
#Profit%=profit/CP*100

df['Profit%']=(df['Profit']/df['CP'])*100

df['Profit%'].head()
df.head(5)
df.iloc[[0,1,2,3,4],[14,20]]
#sort with high perentage

df.sort_values(['Profit%','Product Name'], ascending=False).groupby('Profit%').head()
#LETS LOOK AT THE DATA WRT TO CUSTOMER LEVEL

df['Customer ID'].nunique()
#Top 10 customers wo ordered frequently.

df_top10=df['Customer Name'].value_counts().head(10)

df_top10
fig=plt.figure(figsize=(16,8))

ax=fig.add_subplot()

s=sns.countplot('Segment', data=df)

plt.show()
#to count each segment

fig=plt.figure(figsize=(16,8))

ax=fig.add_subplot(111)

s=sns.countplot('Segment',data=df)

for s in ax.patches:

    ax.annotate('{:0.f}'.format(s.get_height()),(s.get_x()+0.15,s.get_height()+1))

plt.show()
#top 20 customers benefitted from th store

sorttop20=df.sort_values(['Profit'], ascending=False).head(20)

fig=plt.figure(figsize=(16,8))

ax=fig.add_subplot(111)

p=sns.barplot(x='Customer Name', y='Profit', hue='State', palette='Set1', data=sorttop20, ax=ax)

ax.set_title('Top 20 profitable customers')

ax.set_xticklabels(p.get_xticklabels(), rotation=75) #teda label

plt.tight_layout()

plt.show()
#no of unique orders

df['Order ID'].nunique()
#calculte time taken by the order to ship and convert no.of days in int format

df.head(1)
df['Shipment Duration']= pd.to_datetime(df['Ship Date'])-pd.to_datetime(df['Order Date'])

df['Shipment Duration']
df.iloc[:,[0,3,21]]
#Details of cutomer, total products purchased, products they purchased, first and  last purchase date,location from the customer plcsed an roder

#creating function and appending all info it

def agg_customer(x):

    d=[]

    d.append(x['Order ID'].count())

    d.append(x['Sales'].sum())

    d.append(x['Profit%'].mean())

    d.append(pd.to_datetime(x['Order Date']).min())

    d.append(pd.to_datetime(x['Order Date']).max())

    d.append(x['Product Name'].unique())

    d.append(x['City'].unique())

    return pd.Series(d,index=['#Purchases','Total_Sales','Average Profit% Gained','First Purchase Date','Last Purchase Date','Products Purchased','Locations Count'])
df_agg=df.groupby('Customer ID').apply(agg_customer)

df_agg.head(3)
#year of order

df['order year']=df['Order Date'].dt.year

df['order year'].head()
#calculating profit gaines per category

fig=plt.figure(figsize=(16,8))

ax=fig.add_subplot(111)

sns.barplot('order year', 'Profit%', hue='Sub-Category', palette='Paired',data=df)

for o in ax.patches:

    ax.annotate('{:.0f}'.format(o.get_height()), (o.get_x()+0.15, o.get_height()+1))

plt.show()
#sales per year

df.groupby('order year')['Sales','Profit%'].sum().plot.bar()

plt.title("Sales per year")