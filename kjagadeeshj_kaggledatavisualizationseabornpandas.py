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
import matplotlib.pyplot as plt 

import seaborn as sns
os.chdir("../input")

os.listdir()
df = pd.read_csv("superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")

df.shape
df.head(10)
df.shape[0]
df.shape[1]
df.columns.values
df.info()
# 1. Who are the top-20 most profitable customers. Show them through plots.



top20 = df.groupby(["Customer Name"])['Profit'].aggregate(np.sum).reset_index().sort_values('Profit',ascending = False).head(20)

top20
sns.barplot(x="Customer Name", y="Profit", data = top20)
figsize = plt.figure(figsize = (5,5))
ax1 = figsize.add_subplot(111)

sns.countplot(x = "Customer Name",

            data = top20,

            ax = ax1

             )
ax1.set_title("Count plot")
ax1.set_xlabel("CustomerNames")
plt.show()
# Key Observations

# Top 3 customers who has maximum profits are in the order of Tamara Chand, Raymond Buch and Sanjit Chand
# 2. What is the distribution of our customer segment

segcnt = df['Segment'].value_counts().index

segcnt
df.Segment.value_counts()
sns.countplot("Segment", data=df, order = segcnt)
# 3. Who are our top-20 oldest customers

df.dtypes
df['OrdDt'] = pd.to_datetime(df['Order Date'])

df.dtypes
# Observation

# Segment is categorized as Consumer, Corporate Segment, and Home Office.

# Consumer has topped the overall sales with about 26518 sales in total.
# 3. Who are our top-20 oldest customers?
# Sort data by order date in acending grouped by Customer Name

customerbyorddt = pd.DataFrame({'NumOrders' : df.groupby(["OrdDt", "Customer Name"]).size()}).reset_index()

#Display the top 20 oldest customer with number of order placed

customerbyorddt.head(20)
# 4. Which customers have visited this store just once

# Grouped by Customer Name adn get the order count

customerbyorders = pd.DataFrame({'NumOrders' : df.groupby(["Customer Name"]).size()}).reset_index()

#Display customers with only one order

customerbyorders[customerbyorders['NumOrders'] == 1]
# All customers have placed more than 1 order and none have placed only 1 order
#5. Relationship of Order Priority and Profit

df['Order Priority'].value_counts()
sns.boxplot("Order Priority", "Profit", data = df)
# There seem to be no direct relationship between Profit and the Order Priority.

# Still Medium priority Orders have had the highest profit at times depending on the product.
# 6. What is the distribution of customers Market wise?

df['Market'].value_counts()

numcustbymarket = pd.DataFrame({'CustCnt': df.groupby(['Market']).size()}).reset_index()

numcustbymarket
sns.barplot(x = 'Market', y = 'CustCnt', data = numcustbymarket)
# 7. What is the distribution of customers Market wise and Region wise



df['Region'].value_counts()

CustomersByMarketRegion = pd.DataFrame({'Count' : df.groupby(["Market","Region","Customer Name"]).size()}).reset_index()



sns.countplot("Market",

              hue= "Region", 

              data = CustomersByMarketRegion)
# Africa, EMEA and APAC are the top 3 largest markets

# APAC has the most regions that have most sales