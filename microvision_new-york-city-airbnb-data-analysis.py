import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')
# Save filepath to variable for easier access

df_filepath = "../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv"



# Read the data and store data in DataFrame titled airbnb

df = pd.read_csv(df_filepath)
# Preview the dataset

df.head()
# Summary of the statistics in the dataset

df.describe()
df.info()
# Select data subset and rename the dataset

df_features = ['host_id','price', 'room_type','neighbourhood_group','neighbourhood', 

               'latitude', 'longitude', 'minimum_nights', 'availability_365']

airbnb = df[df_features]

airbnb.info()
# Chack missing values

airbnb.isnull().sum()
# check the renamed dataset structure again

airbnb.head(10)
airbnb.info()
# Encode the categorical variables and create new columns

encoder_room = LabelEncoder()

airbnb['roomtype_code'] = encoder_room.fit_transform(airbnb['room_type'])



encoder_ng = LabelEncoder()

airbnb['neighbourhoodGroup_code'] = encoder_ng.fit_transform(airbnb['neighbourhood_group'])



encoder_n = LabelEncoder()

airbnb['neighbourhood_code'] = encoder_n.fit_transform(airbnb['neighbourhood'])
# Check new encoded dataset

airbnb.head(10)
plt.figure(figsize=(10, 7))

sns.heatmap(airbnb.corr(), annot=True, linewidth=0.2, fmt='.2f', cmap='Reds')

plt.show()
plt.figure(figsize=(10, 7))

sns.countplot(x=airbnb['room_type'])

plt.title("Number of Room Types in NYC", fontsize=20)

plt.xlabel("Room Types", fontsize=15)

plt.ylabel("Number of Rooms", fontsize=15)

print(airbnb.room_type.value_counts())    
# In each neighbourhood group

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

sns.countplot(x=airbnb['neighbourhood_group'])

plt.title("Number of Rooms in Neighbourhood Groups", fontsize=20)

plt.xlabel("Neighbourhood Group", fontsize=15)

plt.ylabel("Count", fontsize=15)

    

# Room Types

plt.subplot(1, 2, 2)

sns.countplot(x=airbnb['neighbourhood_group'], hue=airbnb.room_type)

plt.title("Number of Room Types in Neighbourhood Groups", fontsize=20)

plt.xlabel("Neighbourhood Group", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

sns.barplot(x=airbnb['room_type'], y=airbnb['minimum_nights'])

plt.title("Minimum Nights in Room Type", fontsize=20)

plt.xlabel("Room Types", fontsize=15)

plt.ylabel("Minimum Nights (Count)", fontsize=15)

    

plt.subplot(1, 2, 2)

sns.barplot(x=airbnb['neighbourhood_group'], y=airbnb['minimum_nights'])

plt.title("Minimum Nights in Location", fontsize=20)

plt.xlabel("Nighbourhood Group", fontsize=15)

plt.ylabel("Minimum Nights (Count)", fontsize=15)

plt.show()
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x=airbnb['room_type'], y=airbnb['availability_365'])

plt.xlabel("Room Types", fontsize=15)

plt.ylabel("availability_365", fontsize=15)

    

plt.subplot(1, 2, 2)

sns.boxplot(x=airbnb['neighbourhood_group'], y=airbnb['availability_365'])

plt.xlabel("Nighbourhood Group", fontsize=15)

plt.ylabel("availability_365", fontsize=15)

plt.show()
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)

sns.scatterplot(x=airbnb['latitude'], y=airbnb['longitude'], hue=airbnb['neighbourhood_group'])

plt.title('Neighbourhood Group', fontsize=20)

    

plt.subplot(1, 2, 2)

sns.scatterplot(x=airbnb['latitude'], y=airbnb['longitude'], hue=airbnb['room_type'])

plt.legend(loc='lower right')

plt.title('Room Type', fontsize=20)

plt.tight_layout()

plt.show()
plt.figure(figsize=(10, 10))

plt.subplot(3, 1, 1)

sns.distplot(airbnb[airbnb.room_type=='Private room']['price'])

plt.title('Price Distribution of Private Room ', fontsize=20)



plt.subplot(3, 1, 2)

sns.distplot(airbnb[airbnb.room_type=='Entire home/apt']['price'])

plt.title('Price Distribution of Entire Home & Apartment', fontsize=20)



plt.subplot(3, 1, 3)

sns.distplot(airbnb[airbnb.room_type=='Shared room']['price'])

plt.title('Price Distribution of Shared Room', fontsize=20)



plt.tight_layout()

plt.show()
plt.figure(figsize=(10, 20))

plt.subplot(5, 1, 1)

sns.distplot(airbnb[airbnb.neighbourhood_group=='Brooklyn']['price'])

plt.title('Price Distribution in Brooklyn', fontsize=20)



plt.subplot(5, 1, 2)

sns.distplot(airbnb[airbnb.neighbourhood_group=='Manhattan']['price'])

plt.title('Price Distribution in Manhattan', fontsize=20)



plt.subplot(5, 1, 3)

sns.distplot(airbnb[airbnb.neighbourhood_group=='Queens']['price'])

plt.title('Price Distribution in Queens', fontsize=20)



plt.subplot(5, 1, 4)

sns.distplot(airbnb[airbnb.neighbourhood_group=='Staten Island']['price'])

plt.title('Price Distribution in Staten Island', fontsize=20)



plt.subplot(5, 1, 5)

sns.distplot(airbnb[airbnb.neighbourhood_group=='Bronx']['price'])

plt.title('Price Distribution in Bronx', fontsize=20)

plt.tight_layout()

plt.show()