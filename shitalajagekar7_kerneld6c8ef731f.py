import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/listings_summary.csv")
df.shape
format(df.duplicated().sum()) # Check wether duplicate entries are their or not. here 0 represent no duplkicate data is their.
df.columns
columns=['id','space','description','neighborhood_overview','property_type','room_type','accommodates','bathrooms',

         'bedrooms','beds','bed_type','amenities','square_feet','price','security_deposit','cleaning_fee','extra_people',

         'minimum_nights','review_scores_rating','instant_bookable','cancellation_policy']
temp_data=df[columns].set_index('id')
temp_data.head()
temp_data.isnull().sum(axis=0)
temp_data.security_deposit.fillna('$0.00', inplace=True) 
temp_data.cleaning_fee.fillna('$0.00',inplace=True)
temp_data[['security_deposit','cleaning_fee']].isnull().sum(axis=0)
temp_data.head(3)
temp_data.room_type.value_counts()
temp_data.dtypes
temp_data.price=list(map(lambda x: x.replace(',',''),temp_data.price))
temp_data.security_deposit=list(map(lambda x: x.replace(',',''),temp_data.security_deposit))
temp_data.cleaning_fee=list(map(lambda x: x.replace(',',''),temp_data.cleaning_fee))
temp_data['price'].value_counts()
temp_data.price=list(map(lambda x: x.replace('$','0'),temp_data.price))
temp_data.security_deposit=list(map(lambda x: x.replace('$','0'),temp_data.security_deposit))
temp_data.cleaning_fee=list(map(lambda x: x.replace('$','0'),temp_data.cleaning_fee))
temp_data['security_deposit'].value_counts()
temp_data['cleaning_fee'].value_counts()
temp_data['price'].value_counts()
temp_data[['price','security_deposit','cleaning_fee']]=temp_data[['price','security_deposit','cleaning_fee']].astype(float)
temp_data['Total_price']=temp_data[['security_deposit','cleaning_fee','price']].sum(axis=1)
temp_data.head(2)
avg_perRoom=temp_data.groupby(['room_type'])['price'].agg(np.mean)

#data.groupby(['neighbourhood','room_type'])['price'].agg(['mean'])
avg_perRoom.plot(kind='bar', figsize=(8,8), fontsize=10,color='green')

plt.title("Average Price per Room Type",fontsize=15)

plt.xlabel("Room Type",fontsize=12)

plt.ylabel("Average price in dollar",fontsize=12)

plt.show()
temp_data.bed_type.value_counts()
bed_perRoom=temp_data.groupby(['room_type'])['bed_type']
bed_perRoom.value_counts()
property_type_perRoom=temp_data.groupby(['room_type'])['property_type']

                 
property_type_perRoom.value_counts()
score_perRoom=temp_data.groupby(['room_type'])['review_scores_rating'].agg(np.mean)



score_perRoom.plot(kind='pie',autopct="%1.1f%%",startangle=90,figsize=(8,8))
# Cancellation Policy per Room Type
temp_data.cancellation_policy.value_counts()
cancellation_perRoom=temp_data.groupby(['room_type'])['cancellation_policy']
cancellation_perRoom.value_counts()
f,ax = plt.subplots(figsize=(9,9))

sns.heatmap(temp_data.corr(),annot=True,linewidth=.5,fmt ='.2f',ax = ax)