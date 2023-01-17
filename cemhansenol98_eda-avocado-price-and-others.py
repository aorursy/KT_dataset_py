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
data = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')

data.head()
data.info()
data['month'] = data['Date'].apply(lambda x: x.split('-')[1])

data['month'] = data.month.astype('int64')

data.head()
grouped_month = data.groupby('month').mean()

grouped_month.head(15)

import matplotlib.pyplot as plt



months= range(1,13)

plt.bar(grouped_month.index,grouped_month['AveragePrice'])

plt.xticks(months)

plt.show()
grouped_month2 = data.groupby('month').sum()



plt.bar(grouped_month2.index,grouped_month2['AveragePrice'])

plt.xticks(months)

plt.show()
fig, ax1 = plt.subplots()



ax2=ax1.twinx()

ax1.bar(grouped_month2.index,grouped_month2['AveragePrice'],color='g')

ax2.plot(grouped_month.index,grouped_month['AveragePrice'],color='r')



ax1.set_xlabel('Month Number')

ax1.set_ylabel('Sum Average Price',color='g')

ax2.set_ylabel('Mean Average Price',color='r')



plt.show()
import seaborn as sns

plt.figure(figsize=(10,6))

sns.lineplot(x="month", y="AveragePrice", hue='type', data=data)

plt.show()
grouped_month = data.groupby('year')

grouped_month.head()
data.year.unique()
grouped_year = data.groupby('year').mean()

grouped_year2 = data.groupby('year').sum()



years = ['2015',' 2016',' 2017', '2018']



fig, ax1 = plt.subplots()



ax2=ax1.twinx()

ax1.bar(grouped_year2.index,grouped_year2['AveragePrice'],color='g')

ax2.plot(grouped_year.index,grouped_year['AveragePrice'],color='r')



ax1.set_xlabel('Year Number')

ax1.set_ylabel('Sum Average Price',color='g')

ax2.set_ylabel('Mean Average Price',color='r')



plt.show()
grouped_type = data.groupby('type')

grouped_type.head()
grouped_type.mean()['AveragePrice']
grouped_type.mean()['Total Volume']
plt.figure(figsize=(12,10))

sns.barplot(x="AveragePrice",y="region",data= data)
plt.figure(figsize=(12,10))

sns.barplot(x="Total Volume",y="region",data= data)
data2 = data[data.region!='TotalUS']
plt.figure(figsize=(12,10))

sns.barplot(x="Total Volume",y="region",data= data2)
import matplotlib

import squarify



volume_order = data.groupby('region')['Total Volume'].sum(

                        ).sort_values(ascending = False).reset_index()





volume_values = [i for i in range(volume_order.shape[0])]

norm = matplotlib.colors.Normalize(vmin = min(volume_values), vmax = max(volume_values))



plt.figure(figsize = (18, 10))

squarify.plot(sizes = volume_order['Total Volume'], alpha = 0.8,

              label = volume_order.region)

plt.title('Region Total Volume Map', fontsize = 20)

plt.axis('off')

plt.show()


bag_order = data.groupby('region')['AveragePrice'].sum(

                        ).sort_values(ascending = False).reset_index()





bag_values = [i for i in range(bag_order.shape[0])]

norm = matplotlib.colors.Normalize(vmin = min(bag_values), vmax = max(bag_values))





plt.figure(figsize = (20, 12))

squarify.plot(sizes = bag_order['AveragePrice'], alpha = 0.8,

              label = bag_order.region)

plt.title('Region - Average Price Map', fontsize = 20)

plt.axis('off')

plt.show()