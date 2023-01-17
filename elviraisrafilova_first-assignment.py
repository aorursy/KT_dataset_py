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

import shapefile as shp

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

data = pd.read_csv('../input/winemag-data-130k-v2.csv')

data.info()
#data correlation table for numeric values

#there is a slight correlation between points and price of wine

data.corr()
#for visualising correlation heatmap used

f,ax = plt.subplots(figsize=(3, 3))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# displays first 10 rows in dataset

data[:10]
data.head(10)

#code shows the same result as previous
data.columns

#displays column titles existing in dataset
#displays frequency of each country in dataset

data ['country'].value_counts()
# Scatter Plot 

# x = points, y = price

data.plot(kind='scatter', x='points', y='price',alpha = 0.5,color = 'red')

plt.xlabel('points')              # label = name of label

plt.ylabel('price')

plt.title('points price Scatter Plot')            # title = title of plot
#frequency distribution diagram shows the frequency rearding to the countries in the data set. 

#don't find how to arrange x-axis for graph being more readable and presentable.

fig, ax = plt.subplots()

country_count = data['country'].value_counts()

sns.set(style="darkgrid")

sns.barplot(country_count.index, country_count.values,alpha=1)

plt.title('Frequency Distribution of Country')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.show()
#did not find how to arrange figure size!!!



country_count = data['country'].value_counts()

plt.barh (country_count.index, country_count.values, alpha=1)

plt.title('Bar Chart.Country of wine origin')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.show()
#did not find how to arrange frame size, x=axis



data.price.plot(kind = 'hist',bins = 50,figsize = (8,8))

plt.show()
#trying to arrange figure size and x-axis but not succeed

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

x = data['price'].value_counts()

plt.hist(x, bins=100)

plt.gca().set(title='Frequency of prices Histogram', ylabel='Frequency');
#displays frequency of each price in dataset

data ['price'].value_counts()
x = data['price']>3000    

data[x]



 # There is only 1 wine which have higher price value than 3000,

 # so there is something wrong in frequency historam
#wine from Australia has price equals to 820

b = data['price'] == 820    

data[b]
z = data['points']>90     # shows wines which points are higher than 90 over 100

data[z]
#shows the total of wıne types which took more than 90 points

count_z= z.value_counts()

print(count_z)
#wine which point is higher than 90 and price is lower than 15

#there are 174 wine in this category

data[np.logical_and(data['points']>90, data['price']<15)]
series = data['points']        

print(type(series))

data_frame = data[['points']]  

print(type(data_frame))