# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the hotel_bookings.csv file

df_bookings = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df_bookings.head()
# Let us analyze the size of DataFrame

df_bookings.shape
# Get data type for all these 32 features

df_bookings.info()
# Updating missing values

df_bookings.fillna({'children':0, 'country':'Unknown', 'agent':0.0}, inplace=True)



# Drop 'Company' column

#df_bookings.drop('company', axis=1, inplace=True)



df_bookings.head()
len(df_bookings[df_bookings.duplicated()])
# Let's remove 32001 duplicate records from the DataFrame

df_bookings.drop_duplicates(inplace=True)
# Check correlation on the heatmap

fig,axes = plt.subplots(1,1,figsize=(10,7))



sns.heatmap(df_bookings.corr(), cmap='coolwarm', linecolor='white')



plt.show()
df_bookings['hotel'].value_counts()
fig = plt.figure(figsize=(15,5))



plt.hist(df_bookings['lead_time'], bins=40)



plt.xlabel('Lead time')

plt.ylabel('count')

plt.title('How much time guests do the bookings?')



plt.show()
# Next column 'is_canceled'.

df_bookings['is_canceled'].value_counts()
# Also, let's see how many cancellations are there in each hotel

sns.countplot('hotel', data=df_bookings, hue='is_canceled')



plt.show()
df_bookings.groupby('hotel')['is_canceled'].value_counts(normalize=True)*100
# Let us view the average daily price per day per customer. Here, we will consider that the hotels are not charging babies 

sns.boxplot(df_bookings['adr'])



plt.show()
df_bookings.drop(df_bookings[df_bookings['adr']>5000].index, axis=0, inplace=True)

sns.boxplot(df_bookings['adr'])



plt.show()
# Prices round the year

sns.barplot(x='adr', y='arrival_date_month', data=df_bookings, hue='hotel')



plt.show()
fig = plt.figure(figsize=(15,5)) # Create matplotlib figure



plt.hist(df_bookings['stays_in_week_nights'][df_bookings['stays_in_week_nights'] < 10].dropna(), 

         bins=8,alpha = 1,color = 'lemonchiffon',label='Stays in week night' )



plt.hist(df_bookings['stays_in_weekend_nights'][df_bookings['stays_in_weekend_nights'] < 10].dropna(),

         bins=8, alpha = 0.5,color = 'blueviolet',label='Stays in weekend night' )



plt.xlabel('No.of days')

plt.ylabel('Count')

plt.title('No. of Bookings in Week & Weekends')

plt.legend(loc=1)



plt.show()
df_price = df_bookings.groupby('reserved_room_type')['adr'].agg({'Average_price':'mean', 'No. of bookings':'size'})

df_price.reset_index(inplace=True)



df_price
# Let's visualize these to understand better

fig,ax = plt.subplots(1, 2, figsize=(15,5))



# Plot 1 for checking average price per room

ax[0].plot(df_price['reserved_room_type'], df_price['Average_price'], color='red')

ax[0].set_xlabel('Room Types')

ax[0].set_ylabel('Average Price')

ax[0].set_title('Average price per room type')



# Plot 2 for checking number of bookings

ax[1].plot(df_price['reserved_room_type'], df_price['No. of bookings'], color='green')

ax[1].set_xlabel('Room Types')

ax[1].set_ylabel('No. of bookings')

ax[1].set_title('Number of booking for each room')



plt.show()
df_bookings['market_segment'].value_counts()
# Let's visualize the market segment in a pie chart

fig = plt.figure(figsize=(10,10))



market_size = df_bookings['market_segment'].value_counts().tolist()

labels = df_bookings['market_segment'].value_counts().index.tolist()



plt.pie(market_size, labels=labels, autopct='%1.1f%%', startangle=90)



plt.show()
# Check count of Agents 

agent_list = list(df_bookings['agent'].value_counts().index)



print('Total number of agents in list: ', len(agent_list))
# Let's see which are the top 10 agents who are responsible for booking in these hotels

fig = plt.figure(figsize=(10,10))



# Values to be seen

bookings = df_bookings['agent'].value_counts().tolist()

agent_list = df_bookings['agent'].value_counts().index.tolist()



# Pop out top 3 agents with maximum bookings

explode = (0.10,0.07,0.03,0,0,0,0,0,0,0)



plt.pie(bookings[:10], labels=agent_list[:10], explode=explode, autopct='%1.1f%%', startangle=90)

plt.tight_layout()

plt.title('Best agent')



plt.show()
# Print the top 10 count

fig = plt.figure(figsize=(15,5))



x = df_bookings['country'].value_counts().index[:10]

y = df_bookings['country'].value_counts()[:10]



plt.bar(x,y, color='green')

plt.xlabel('Countries')

plt.ylabel('Customer count')

plt.title('Top 10 customer count')



plt.show()
# Check which type of customers are visiting in each hotel



sns.countplot(df_bookings['customer_type'], hue=df_bookings['hotel'], order=df_bookings['customer_type'].value_counts().index)



plt.show()