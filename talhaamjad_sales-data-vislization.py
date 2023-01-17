# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import pylab as pl

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
import pandas.plotting

from IPython import display

from ipywidgets import interact, widgets

import re

import mailbox

import csv

import scipy.stats
df = pd.read_csv("../input/sales-dataset/Sales.csv")

df
df.describe()
data = df['SALES']

data
plt.scatter(df.ORDERDATE, df.SALES,  color='blue')

plt.xlabel("Years")

plt.ylabel("ORDERDATE")

plt.show()
histogram = df[['SALES','QUANTITYORDERED']]

histogram.hist()

plt.show()
plt.scatter(df.ORDERDATE, df.SALES,  color='blue')

plt.xlabel("ORDERDATE")

plt.ylabel("SALES")

plt.show()
x = df['ORDERDATE'].head()

y = df['SALES'].head()

plt.bar(x,y)
cf = df['YEAR_ID']

a = np.unique(cf)

a
plt.ylabel("SALES")

y.plot()

z = df['QUANTITYORDERED'].head()

plt.ylabel("Quantity")

z.plot()
pd.DataFrame(df.YEAR_ID.value_counts())
pie_chart = pd.DataFrame(df.PRODUCTLINE.value_counts(normalize=True))

pie_chart
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

pie_charts = np.unique(df['PRODUCTLINE'])

ax.pie(pie_chart,labels = pie_charts,autopct='%1.2f%%')

plt.show()
dealsize = pd.DataFrame(df.DEALSIZE.value_counts(normalize=True))

dealsize


fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

deal_size = np.unique(df['DEALSIZE'])

ax.pie(dealsize,labels = deal_size,autopct='%1.2f%%')

plt.show()

#data['Revenue'].sum()

add = df['SALES'].sum()

print("Total Sales is",add)

               

#numpy.where(data['Revenue']<data['Expense'],1,2)
yearly = pd.DataFrame(df.YEAR_ID.value_counts(normalize=True))

yearly
smoking = df.groupby("PRODUCTLINE").YEAR_ID.value_counts()

smoking
smoking.index
smoking.unstack()
plt.figure(figsize=(10,4))



plt.subplot(1,2,1);df.YEAR_ID.value_counts().plot(kind='barh', color=['C0', 'C1']); plt.title('YEAR')

plt.subplot(1,2,2);df.PRODUCTLINE.value_counts().plot(kind='barh', color=['C2', 'C3']); plt.title('smoker')
smoking.unstack().plot(kind='bar', stacked =True)
smoking.plot(kind = 'bar')
sales = df.groupby("YEAR_ID").DEALSIZE.value_counts()

sales
sales.index
sales.unstack()
plt.figure(figsize=(10,4))



plt.subplot(1,2,1);df.DEALSIZE.value_counts().plot(kind='barh', color=['C0', 'C1']); plt.title('DEALSIZE')

plt.subplot(1,2,2);df.DEALSIZE.value_counts().plot(kind='barh', color=['C2', 'C3']); plt.title('DEALSIZE')
sales.unstack().plot(kind='bar', stacked =True)
country = df.groupby("YEAR_ID").COUNTRY.value_counts()

country
country.index
country.unstack()
plt.figure(figsize=(10,4))



plt.subplot(1,2,1);df.COUNTRY.value_counts().plot(kind='barh', color=['C0', 'C1']); plt.title('COUNTRY')

plt.subplot(1,2,2);df.COUNTRY.value_counts().plot(kind='barh', color=['C2', 'C3']); plt.title('COUNTRY')
country.unstack().plot(kind='bar', stacked =True)
df1 = pd.DataFrame(df) 

  

name = df1['YEAR_ID'].head(50) 

price = df1['SALES'].head(50) 



# Figure Size 

fig = plt.figure(figsize =(10, 7)) 

  

# Horizontal Bar Plot

#plt.xticks(x_pos, name,rotation=60)

plt.bar(name, price) 

plt.show() 
df2 = pd.DataFrame(df) 

  

name1 = df2['PRODUCTLINE'] 

price1 = df2['SALES'] 

  

# Figure Size 

fig = plt.figure(figsize =(10, 7)) 

  

# Horizontal Bar Plot

#plt.xticks(name1,rotation=60)

plt.bar(name1, price1) 

plt.show() 
df3 = pd.DataFrame(df) 

  

name2 = df3['QUANTITYORDERED'].head(100)

price2 = df3['MONTH_ID'].head(100) 

  

# Figure Size 

fig = plt.figure(figsize =(10, 7)) 

#x_pos = np.arange(len(price2))

 

# Horizontal Bar Plot

plt.bar(price2, name2) 

#plt.xticks(x_pos,price2,rotation=90)

plt.show() 
df['year'] = pd.DatetimeIndex(df['ORDERDATE']).year

df['year']
df4 = pd.DataFrame(df) 

  

name4 = df4['year'].head(50) 

price4 = df4['SALES'].head(50) 

  

# Figure Size 

fig = plt.figure(figsize =(10, 7)) 

#plt.scatter(name4, price4)

# Horizontal Bar Plot 

plt.bar(name4, price4)

plt.show() 