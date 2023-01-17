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

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame

%matplotlib inline
# With the provided airbnb information, try to study how to invest and manage rental properties in New York City.

# To find out the most popular locations, price range and room types are very important to keep the listing properties busy.

data = pd.read_csv(r'/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

print(data.info())

print(data.isnull().sum())

data.head()



# exploring data: checking the column name and columns with null values

# showing the first several rows of the data set
sns.scatterplot(x='latitude', y='longitude', data=data, hue='neighbourhood_group')

plt.title('Locations of Airbnb listing in New York City')



# showing all the airbnb listing locations in the New York City

# different color represent different neighbourhood
df1 = data['id'].groupby(data['neighbourhood_group']).count()

df1 = df1.to_frame().reset_index()

df1 = df1.rename(columns = {'id':'total number of listing per group'})

print(df1)

# plt.bar(df1['neighbourhood_group'], df1['total number of listing per group'], align='center', alpha=0.8, color='g')

# plt.ylabel('Total number of listing per neighbourhood_group')

# plt.xlabel('Neighbourhood_group')

# plt.title('The most popular area for Airbnb in New York City')



plt.pie( df1['total number of listing per group'], labels=df1['neighbourhood_group'],shadow=True, startangle=200,autopct='%1.1f%%')



# a pie plot showing which neighbourhood has the most and least airbnb listing in New York City
a = data[data['neighbourhood_group']=='Manhattan']['availability_365'].sum()/len(data[data['neighbourhood_group']=='Manhattan'])

b = data[data['neighbourhood_group']=='Bronx']['availability_365'].sum()/len(data[data['neighbourhood_group']=='Bronx'])

c = data[data['neighbourhood_group']=='Brooklyn']['availability_365'].sum()/len(data[data['neighbourhood_group']=='Brooklyn'])

d = data[data['neighbourhood_group']=='Queens']['availability_365'].sum()/len(data[data['neighbourhood_group']=='Queens'])

e = data[data['neighbourhood_group']=='Staten Island']['availability_365'].sum()/len(data[data['neighbourhood_group']=='Staten Island'])

# print(a,b,c,d,e)

days = {'Neibourhood_group': ['Manhattan','Bronx','Brooklyn','Queens','Staten Island'],'Number of days available': [a,b,c,d,e]}

df2 = DataFrame(days, columns=('Neibourhood_group', 'Number of days available'))

plt.bar(df2['Neibourhood_group'], df2['Number of days available'], align='center', color='g', alpha=0.8)

plt.ylabel('Average vacancy days in 2019')

plt.xlabel('Neighbourhood_group')

plt.title('The most competitive neighbours in New York City')



# the most competitive neighbourhood is Staten Island, since the average listing vacancy is the longest.

# Manhattan and Brooklyn should be idea location for new airbnb properties investors, because there are less availables in these two neighbourhood

a = data[(data['price']<500)]['availability_365'].mean()

b = data[(data['price']>500)&(data['price']<1000)]['availability_365'].mean()

c = data[(data['price']>1000)&(data['price']<2000)]['availability_365'].mean()



a1 = data[data['price']<500]['price'].mean()

b1 = data[(data['price']<1000)&(data['price']>500)]['price'].mean()

c1 = data[(data['price']<2000)&(data['price']>1000)]['price'].mean()



average_availability = [a,b,c]

average_price = [a1,b1, c1]

plt.plot(average_price,average_availability, 'b.--', markersize=10, marker='s')



plt.xlabel('Price')

plt.ylabel('Average vacancy days')

plt.title('Sensitivity of prices on vacancy days')



# Price is another very important factor for attracting potential customers 

# From the plot, it seems price after a centain range will increase the listing vacancy dramatically. So the relative low range prcie will

# be suggested for reducing the longer vacancy risk
data.groupby('room_type').count()



a = data[data['room_type']=='Entire home/apt']['availability_365'].mean()

b = data[data['room_type']=='Private room']['availability_365'].mean()

c = data[data['room_type']=='Shared room']['availability_365'].mean()

x = ['Entire home/apt','Private room','Shared room']

y = [a,b,c]

plt.bar(x, y, align='center', color='b', alpha=0.8)

plt.xlabel('Room type')

plt.ylabel('Average vacancy days')

plt.title('Most needed room type in New York City')



# Entire home or private room listing are more needed than shared room in New York City
a=data['number_of_reviews']

b=data['availability_365']

c=data[data['calculated_host_listings_count']==5]['availability_365'].mean()

d=data[data['calculated_host_listings_count']==6]['availability_365'].mean()

print(c,d)



from collections import Counter

names=[]

words=[]

for name in data['name']:

    a = str(name).split()

    names.append(a)

    

for name in names:

    for word in name:

        word=word.lower()

        words.append(word)

top_20 = Counter(words).most_common()

top_20 = top_20[0:20]



df3 = pd.DataFrame(top_20)

df3 = df3.rename(columns ={0:'most popular word', 1:'count'})

viz = sns.barplot(df3['most popular word'], df3['count'])

viz.set_xticklabels(viz.get_xticklabels(),rotation=80)

viz.set_title('The top 20 popular words for listing names')



# the top 20 key words shown in the plot

# Words like 'private', 'cozy' and 'spacious' seems most popular words for decribing listing properties
plt.subplots(figsize=(15,6))

plt.subplot(1,2,1)

sns.scatterplot(x='latitude', y='longitude', data=data, hue='neighbourhood_group')

plt.title('Locations of Airbnb listing in New York City')



plt.subplot(1,2,2)

plt.pie( df1['total number of listing per group'], labels=df1['neighbourhood_group'],shadow=True, startangle=200,autopct='%1.1f%%')


plt.subplots(figsize=(15,12))



plt.subplot(2,2,1)

plt.bar(df2['Neibourhood_group'], df2['Number of days available'], align='center', color='g', alpha=0.8)

plt.ylabel('Average vacancy days in 2019')

plt.xlabel('Neighbourhood_group')

plt.title('The most competitive neighbours in New York City')



plt.subplot(2,2,2)

plt.plot(average_price,average_availability, 'b.--', markersize=10, marker='s')

plt.xlabel('Price')

plt.ylabel('Average vacancy days')

plt.title('Sensitivity of prices on vacancy days')



plt.subplot(2,2,3)

plt.bar(x, y, align='center', color='b', alpha=0.8)

plt.xlabel('Room type')

plt.ylabel('Average vacancy days')

plt.title('Most needed room type in New York City')



plt.subplot(2,2,4)

viz = sns.barplot(df3['most popular word'], df3['count'])

viz.set_xticklabels(viz.get_xticklabels(),rotation=80)

viz.set_title('The top 20 popular words for listing names')
