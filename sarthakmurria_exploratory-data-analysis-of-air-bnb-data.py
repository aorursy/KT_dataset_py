import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline
# read_csv() is an inbuilt- function to read the comma separated value file which lies within panda module.



df = pd.read_csv('../input/AB_NYC_2019.csv')
df.head(15)
df.info(memory_usage=True, verbose = True)
df.isnull()
sns.heatmap(df.isnull(), yticklabels=False,cbar=True, cmap='cubehelix')
df['last_review']
df['last_review'].fillna('No Review', inplace=True)

print(df['last_review'])
df['reviews_per_month']
df['reviews_per_month'].isnull()

df['reviews_per_month'].fillna('0.0', inplace=True)

print(df['reviews_per_month'])

df['reviews_per_month'].isnull().sum()
neighbourhood_group = df['neighbourhood_group'].value_counts()
neighbourhood_group.plot.bar(edgecolor='black')
price = df['price']

room_type = df['room_type']
plt.scatter(price, room_type, c='r', edgecolors='black', linewidth=0.19)

plt.style.use('dark_background')

plt.style.context('dark_background')

plt.xlabel('Price Of Room',fontsize=11)

plt.ylabel('Type Of Room',fontsize=11)

plt.title("Comparing Type Of Room On The Basis Of Price",fontsize=13)

# Mean of price column is not that high.

np.mean(df['price'])



# Default index

# It ranges from 0:48895 In a sequential manner

# But we don't know the index of a xyz value we are going to search so that's the reason we need to change the Index

df.index
df.set_index('room_type', inplace=True)
df.head()
enitre_home_min_night = df.loc['Entire home/apt', 'minimum_nights']

print('An Average of Enitre home is '+str(enitre_home_min_night.mean())+' nights')



priv_room_min_night = df.loc['Private room', 'minimum_nights']

print('An average of Private room is '+str(priv_room_min_night.mean())+' nights')



share_room_min_night = df.loc['Shared room','minimum_nights']

print('An average of Shared room is '+str(share_room_min_night.mean())+' nights')
# dom_state_entire_room(Domination state of entire room \)



dom_state_enitre_room = df.loc['Entire home/apt', 'neighbourhood_group']

dom_state_enitre_room.value_counts()

dom_state_enitre_room.value_counts().plot.bar(color='White')
# dom_state_private_room(Dominiation of the state in the private room)



dom_state_private_room = df.loc['Private room','neighbourhood_group']

dom_state_private_room.value_counts()

dom_state_private_room.value_counts().plot.bar(color='yellow')
# dom_state_shared_room (domination of the state in the Shared room spectrum)



dom_state_shared_room = df.loc['Shared room', 'neighbourhood_group']

dom_state_shared_room.value_counts()

dom_state_shared_room.value_counts().plot.bar(color='red')
neighbourhood = df['neighbourhood']



neighbourhood_value_count = neighbourhood.value_counts().head(10).plot.pie(autopct='%1.0f%%', shadow=True)