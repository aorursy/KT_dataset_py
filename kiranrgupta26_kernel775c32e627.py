# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

dataset = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



top_host=dataset.host_id.value_counts().head(10)
neighbor_group = dataset.neighbourhood_group.unique()
rooms_type = []

avg_room_price=[]

private_rooms_avg_price=[]

shared_rooms_avg_price=[]

host_listing_count=[]

number_of_reviews=[]
for group in neighbor_group:

    neighbour = dataset.loc[dataset['neighbourhood_group']==group]

    roomtype = neighbour.room_type.unique()

    private_rooms_avg_price.append(neighbour.loc[neighbour['room_type']=='Private room'].price.mean())

    shared_rooms_avg_price.append(neighbour.loc[neighbour['room_type']=='Shared room'].price.mean())

    host_listing_count.append(neighbour.calculated_host_listings_count.value_counts().sum())

    number_of_reviews.append(neighbour.number_of_reviews.sum())

    for type in roomtype:

        numberof_room_types = neighbour.loc[neighbour['room_type']==type]

        rooms_type.append(type)

        count = numberof_room_types.room_type.value_counts()

        mean_price = numberof_room_types.price.mean()

        avg_room_price.append(mean_price)

        avg_stay = numberof_room_types.minimum_nights.mean()

        

        avg_avaiable =  numberof_room_types.availability_365.mean()

        print( group, ' has ' , type ,' ',count.values ,' with Average Price ',mean_price)

        print('Average Minimum stay ' ,avg_stay)

        print('It is available approx ',avg_avaiable ,'days a year')

    print(' ') 
plt.xlabel('Room Type')

plt.ylabel('Avg Price')

plt.plot(rooms_type,avg_room_price,'bo')

plt.show()



plt.xlabel('Neighbourhood_group')

plt.ylabel('Avg Price for Private Room')

plt.plot(neighbor_group,private_rooms_avg_price,'go')

plt.show()



plt.xlabel('Neighbourhood_group')

plt.ylabel('Avg Price for Shared Room')

plt.plot(neighbor_group,shared_rooms_avg_price,'ro')

plt.show()



plt.xlabel('Neighbourhood_group')

plt.ylabel('Rooms Count Listed on AirBnB')

plt.plot(neighbor_group,host_listing_count,'ro')

plt.show()



plt.xlabel('Neighbourhood_group')

plt.ylabel('No. of reviews')

plt.plot(neighbor_group,number_of_reviews,'ro')

plt.show()