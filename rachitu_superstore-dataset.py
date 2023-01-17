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
#0 Clear Memory

%reset -f



#1 Calling libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#2 Set working directory and read file



print(os.getcwd())

os.listdir()

os.getcwd()

pd.options.display.max_columns = 300

ss = pd.read_csv("../input/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")
#3 Read data



ss.shape
# List top 5 items

ss.head()
#List last 5 items

ss.tail()
#List of all coloumns in the file

ss.columns
# Type of data



ss.dtypes
ss.isnull().sum() #is Null
ss.info #information on data
#4 Top-20 most profitable customers.

# Tamara Chand has the highest Profits.

plt.figure(figsize=(12,8))

top20profit = ss.sort_values('Profit', ascending=False)

top20 = top20profit.head(20)

top20[['Customer Name', 'Profit']]

sns.barplot(x = "Customer Name", y= "Profit", data=top20)  # plotting of top 20 profitable customers

plt.show()



# 5 Distribution of our customer segment

df = plt.figure(figsize=(16,8))

ss['Segment'].value_counts().plot.bar()

plt.title('Sales - Segment-Wise')

plt.ylabel('Count')

plt.xlabel('Segments')

plt.tight_layout()

plt.show()
#6 Who are our top-20 oldest customers



ss['Order Date'] = pd.to_datetime(ss['Order Date'])      

old20Cust= ss.sort_values(['Order Date'], ascending=False).head(20)    

old20Cust.loc[:,['Customer Name']]
#7 Relationship of Order Priority and Profit

ss['Order Priority'].value_counts()



sns.boxplot(

            "Order Priority",

            "Profit",

             data= ss

             )
#8 Which customers have visited this store just once #7 customers

Visit=ss.groupby('Customer ID').apply(lambda x: pd.Series(dict(visit_count=x.shape[0])))

Visit.loc[(Visit.visit_count==1)]
#9 Top 10 Products in Sales

plt.figure(figsize=(16,8))

top10pname = ss.groupby('Product Name')['Row ID'].count().sort_values(ascending=False)

top10pname = top10pname [:10]

top10pname.plot(kind='bar', color='Purple')

plt.title('Top 10 Products in Sales')

plt.ylabel('Count')

plt.xlabel('Products')

plt.show()



#Largest selling product is staples
#10 Region-wise sales    # Central Region tops in  Sales

plt.figure(figsize=(12,8))

ss['Region'].value_counts().plot.bar()

plt.title('Region Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Regions')

plt.show()

#11 What is the distribution of customers Market wise?      #APAC has largest customers

plt.figure(figsize=(12,8))

sns.countplot("Market",data = ss)

plt.title('Market wise distribution of customers')

plt.show()
#12 Top 10 Cities in Sales   # New York City has highest Sales

plt.figure(figsize=(12,8))

top10city = ss.groupby('City')['Row ID'].count().sort_values(ascending=False)

top10city = top10city [:10]

top10city.plot(kind='bar', color='green')

plt.title(' Sales - Top 10 Cities in')

plt.ylabel('Count')

plt.xlabel('Cities')

plt.show()
