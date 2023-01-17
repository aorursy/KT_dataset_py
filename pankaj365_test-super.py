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
""" Analysing Super Store Data

Exercise 3

1. Who are the top-20 most profitable customers. Show them through plots.

2. What is the distribution of our customer segment

3. Who are our top-20 oldest customers

4. Which customers have visited this store just once

5. Relationship of Order Priority and Profit

6. What is the distribution of customers Market wise?

7. What is the distribution of customers Market wise and Region wise"""
%reset -f

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

import os

data = pd.read_csv("../input/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")
# 1. Who are the top-20 most profitable customers. Show them through plots.

top20Cust= data.sort_values(['Profit'], ascending=False).head(20)

sns.barplot(x = "Customer ID",     

            y= "Profit",    

            data=top20Cust)
#2. What is the distribution of our customer segment

sns.catplot(x="Segment",       

            col="Market",      

            data=data,

            kind="count"

            )
#3. Who are our top-20 oldest customers



data['Order Date'] = pd.to_datetime(data['Order Date'])      

old20Cust= data.sort_values(['Order Date'], ascending=False).head(20)

top20Cust.loc[:,['Customer Name']]

#4. Which customers have visited this store just once



Visit=data.groupby('Customer ID').apply(lambda x: pd.Series(dict(visit_count=x.shape[0])))

Visit.loc[(Visit.visit_count==1)]

#5. Relationship of Order Priority and Profit

sns.barplot(x = "Order Priority",     

            y= "Profit",    

            data=data)
#6. What is the distribution of customers Market wise?



ascending_order = data['Market'].value_counts().index

sns.countplot(x="Market", data=data, order = ascending_order, palette="Greens_d")



"""

The EU, LATAM and US have nearly the same market size, 

whereas Africa and EMEA are comparable. The Cnadaian market is miniscule.

"""
#7. What is the distribution of customers Market wise and Region wise



fig = plt.figure(figsize = (25,10))

ax1 = fig.add_subplot(111)

Market1=data[data.Market=='APAC']

sns.countplot("Region", data = Market1, ax = ax1)

ax1.set_xlabel("APAC")



ax2 = fig.add_subplot(121)

Market2=data[data.Market=='LATAM']

sns.countplot("Region", data = Market2, ax = ax2)

ax2.set_xlabel("LATAM")



ax3 = fig.add_subplot(211)

Market3=data[data.Market=='EU']

sns.countplot("Region", data = Market3, ax = ax3)

ax3.set_xlabel("EU")



ax4 = fig.add_subplot(222)

Market4=data[data.Market=='US']

sns.countplot("Region", data = Market4, ax = ax4)

ax4.set_xlabel("US")



ax5 = fig.add_subplot(311)

Market5=data[data.Market=='EMEA']

sns.countplot("Region", data = Market5, ax = ax5)

ax5.set_xlabel("EMEA")



ax6 = fig.add_subplot(322)

Market6=data[data.Market=='Africa']

sns.countplot("Region", data = Market6, ax = ax6)

ax6.set_xlabel("Africa")



ax7 = fig.add_subplot(411)

Market7=data[data.Market=='Canada']

sns.countplot("Region", data = Market7, ax = ax7)

ax7.set_xlabel("Canada")
