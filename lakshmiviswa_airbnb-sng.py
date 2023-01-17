

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



listings = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')
listings.head()
listings.columns
drop_sng = ["id","name","host_id","host_name","last_review","reviews_per_month"]
#dropping the unnecessary columns from the dataframe

listings.drop(drop_sng,inplace=True,axis=1)
#Displaying the columns in the dataframe after dropping few columns

listings.columns
#checking the null values in dataframe

listings.isnull().sum()
import matplotlib                  

import matplotlib.pyplot as plt

import seaborn as sns   

sns.countplot(listings["neighbourhood_group"], linewidth=100)

#sns.countplot(listings["price"], linewidth=200)

fig, pl_ax = plt.subplots(figsize=(10,7))

listings['price'].plot(ax=pl_ax)

#sns.distplot(listings['price'], hist=True, rug=True);
#Reference: https://scentellegher.github.io/programming/2017/07/15/pandas-groupby-multiple-columns-plot.html

#Maximum price charged in each of the regions based on the type of airbnb.

fig, pl_ax = plt.subplots(figsize=(10,7))

listings.groupby(['neighbourhood_group','room_type']).max()['price'].unstack().plot(ax=pl_ax)
#This pivot table helps to look at the analysis of maximum price charged in neighbourhood_groups in numbers and then we can visualise the pivot table in chart.



price_pivot = listings.pivot_table(index='neighbourhood_group', columns='room_type', values='price', aggfunc='max')

price_pivot.head()
#Visualising the pivot table data in a bar chart

price_pivot.plot(kind='bar', figsize=[10,5], stacked=True, colormap='autumn')
#grouping the neighbourhoods based on the hosting list

listings.groupby('neighbourhood')['calculated_host_listings_count'].count().plot(kind='bar', figsize = (10,8))

#df2.plot(figsize=(15,8))

plt.xticks(rotation=100)

plt.show()

#Neighbourhood Kallang has more host listing

group=listings.groupby('neighbourhood_group')['calculated_host_listings_count'].count()

print(group)

listings.groupby('neighbourhood_group')['calculated_host_listings_count'].count().plot(kind='barh', figsize = (10,5), color='brown')

#df2.plot(figsize=(10,5))

plt.show()
#Plotting latitude and longitude values of neighbourhood

plt.figure(figsize=(10,10))

sns.scatterplot(x=listings['longitude'], y=listings['latitude'], hue=listings['neighbourhood_group'])

plt.ioff()
#Central Region with maximum Airbnb listings

g = listings[listings['neighbourhood_group'] == 'Central Region']

plt.figure(figsize=(10,5))

sns.scatterplot(x=g['longitude'], y=g['latitude'], hue=g['neighbourhood_group'])

plt.ioff()

#North Region with minimum Airbnb listings

g1 = listings[listings['neighbourhood_group'] == 'North Region']

plt.figure(figsize=(10,5))

sns.scatterplot(x=g1['longitude'], y=g1['latitude'], hue=g1['neighbourhood_group'])

plt.ioff()
# Finding number of reviews for each neighbourhood

listings.groupby('neighbourhood')['number_of_reviews'].count().plot(kind='bar', figsize = (10,8), color = 'green')

#df2.plot(figsize=(20,10))

plt.show()

listings.groupby('neighbourhood')['price'].max().plot(kind='bar', figsize = (10,8))

#df2.plot(figsize=(20,10))

plt.show()

#Price based on neighbourhood shows the maximum price appears in kallang, Bukit Panjang and Tuas neighbourhood
room_size = listings.groupby(['room_type']).size()

print(room_size)

listings.groupby(['room_type']).size().plot(kind='bar', color = 'brown')

Total_Room_count = listings['room_type'].count()

Total_Room_count
#of total room count, what is the percentage of each room type utilised....



Total_listing=room_size/Total_Room_count*100

Total_listing
#https://pythonspot.com/matplotlib-pie-chart

import matplotlib.pyplot as plt

labels = 'Entire home/apt', 'Private room', 'Shared room'

sizes = [52.257493, 42.759580, 4.982927]

colors = ['gold', 'yellowgreen', 'lightcoral']

explode = (0.1, 0,0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()