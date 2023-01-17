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
data = pd.read_csv("../input/vgsales.csv")

data.head()

data.info()
data.corr()

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.4f',ax=ax)

plt.show()
data.plot(kind='scatter', x='Year', y="Global_Sales",alpha = .8,color = 'blue',figsize= (6,6))

plt.legend()

plt.ylabel('Global_Sales')             

plt.xlabel("Year")

plt.title('Scatter Plot') 

plt.show()
data.plot(kind='scatter', x='NA_Sales', y="Global_Sales",alpha = .8,color = 'g',figsize= (6,6))

plt.legend()

plt.ylabel('Global_Sales')             

plt.xlabel("Year")

plt.title('Scatter Plot') 

plt.show()
ax = plt.gca()



data.plot(kind='line', x = "NA_Sales",y = "Global_Sales", color = "green", ax=ax,grid = True,figsize = (7,7))

data.plot(kind='line', x = "EU_Sales",y = "Global_Sales", color = 'red', ax=ax,grid = True)

data.plot(kind='line', x = "JP_Sales",y = "Global_Sales", color = 'b', ax=ax,grid = True)

data.plot(kind='line', x = "Other_Sales",y = "Global_Sales", color = 'y', ax=ax,grid = True)

plt.legend(loc = "upper left")

plt.show()
data.Year.plot(kind = 'hist',bins = 60,figsize = (8,8))

plt.show()
data.Year.plot(kind = 'hist', bins = 100,figsize = (8,8),cumulative = True)

plt.savefig('graph.png')

plt.show()
pd.Series(data.Platform).value_counts().plot('bar', grid = True , figsize = (14,14))

plt.show()
x = data['Year']>2010     

data[x].mean()
data["Sales_Level"] = ["high" if i > 40 else "low" for i in data.Global_Sales]
data.describe()
data.tail()
print(data['Year'].value_counts(dropna =False))
data.boxplot(column='Rank',by = 'Year')

plt.show()
data_new = data.head(7)    # I only take 5 rows into new data

data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Global_Sales','Sales_Level','Publisher'])

melted
data1 = data['Name'].head()

data2= data['Genre'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data['Platform'] = data['Platform'].astype('category')

data.info()
datatime = data['Year']

datatime_obj = pd.to_datetime(datatime)

dataY = data

dataY.Year = datatime_obj

dataY = dataY.set_index("Year")

dataY.head()