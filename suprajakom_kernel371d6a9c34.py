# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #data visulalization 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load the data - use proper encoding

df = pd.read_csv("../input/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")
df.shape
df.head()

df.columns

df.dtypes
# Who are the most profitable customers

# Customers are not unique, therefore group by customers, add their profits, sort them in descending, and plot the first 20 of them.



top20 = df.groupby(["Customer Name"])['Profit'].aggregate(np.sum).reset_index().sort_values('Profit',ascending = False).head(20)
top20.head

type(top20)

top20.shape

top20

ax1=sns.barplot(x = "Customer Name",y= "Profit",data=top20)

plt=ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)
df.Segment.value_counts()
sns.countplot("Segment",data=df)
# Order date field decides how old our customer is

oldest = pd.DataFrame({'Count' : df.groupby(["Order Date","Customer Name"]).size()}).reset_index()

oldest.head(20)
customer_once = pd.DataFrame({'Count' : df.groupby(["Customer ID"]).size()}).reset_index()

customer_once
customer_once[customer_once['Count'] == 1]
sns.barplot(x = "Order Priority",     

            y= "Profit",    

            data=df)
ascending_order = df['Market'].value_counts().index

sns.countplot(x="Market", data=df, order = ascending_order)
df['Region'].value_counts()

Customers_market_region = pd.DataFrame({'Count' : df.groupby(["Market","Region","Customer Name"]).size()}).reset_index()



sns.countplot("Market",        # Variable whose distribution is of interest

              hue= "Region",    # Distribution will be region-wise

              data = Customers_market_region)