import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df.head()
df.shape
df.isnull().sum()
#visualize missing data

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(df.isnull(),cmap='Blues', yticklabels=False, cbar=False)

plt.show()
df = df.drop(columns = ['agent', 'company'])
#drops rows containing missing data in 'country' column

df = df.dropna(axis=0)
df.isnull().sum()
#Overview of type of hotel



#Enlarging pie chart

plt.rcParams['figure.figsize'] = 9,9



#Indexing labels

labels = df['hotel'].value_counts().index.tolist()



#Convert value counts to list

sizes = df['hotel'].value_counts().tolist()



#Explode to determine how much each section is separated from each other

explode = (0,0.075)



#Coloring pie chart

colors = ['0.75', 'maroon']



plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
df.head()
#Grouping by adults to get summary statistics on hotel type

df['adults'].groupby(df['hotel']).describe()
#Grouping by children to get summary statistics on hotel type

df['children'].groupby(df['hotel']).describe()
#analyzing canceled bookings data

df['is_canceled'] = df.is_canceled.replace([1,0], ['canceled', 'not_canceled'])

canceled_data = df['is_canceled']

sns.countplot(canceled_data)
#Analyzing cancellation rate amongst hotel types

lst1 = ['is_canceled', 'hotel']

type_of_hotel_canceled = df[lst1]

canceled_hotel = type_of_hotel_canceled[type_of_hotel_canceled['is_canceled'] == 'canceled'].groupby(['hotel']).size().reset_index(name='count')

sns.barplot(data = canceled_hotel, x = 'hotel', y = 'count').set_title('Graph depicting cancellation rates in city and resort hotel')
#Graph arrival year

lst3 = ['hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']

period_arrival = df[lst3]



sns.countplot(data = period_arrival, x = 'arrival_date_year', hue = 'hotel').set_title('Graph showing number of arrivals per year')

#Graph arrival month

plt.figure(figsize=(20,5)) # adjust the size of the plot



sns.countplot(data = period_arrival, x = 'arrival_date_month', hue = 'hotel', order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December']).set_title('Graph showing number of arrivals per month',fontsize=20)

plt.xlabel('Month') # Creating label for xaxis

plt.ylabel('Count') # Creating label for yaxis
#Graph arrival dates

plt.figure(figsize=(15,5))



sns.countplot(data = period_arrival, x = 'arrival_date_day_of_month', hue = 'hotel').set_title('Graph showing number of arrivals per day', fontsize = 20)
#Graphing weekend vs. weekday data

sns.countplot(data = df, x = 'stays_in_weekend_nights').set_title('Number of stays on weekend nights', fontsize = 20)
sns.countplot(data = df, x = 'stays_in_week_nights' ).set_title('Number of stays on weekday night' , fontsize = 20)
#Graphing data by types of visitors

sns.countplot(data = df, x = 'adults', hue = 'hotel').set_title("Number of adults", fontsize = 20)
sns.countplot(data = df, x = 'children', hue = 'hotel').set_title("Number of children", fontsize = 20)
sns.countplot(data = df, x = 'babies', hue = 'hotel').set_title("Number of babies", fontsize = 20)
#Graphing booking data by country of origin

country_visitors = df[df['is_canceled'] == 'not_canceled'].groupby(['country']).size().reset_index(name = 'count')



# We will be using Plotly.express to plot a choropleth map. Big fan of Plotly here!

import plotly.express as px



px.choropleth(country_visitors,

                    locations = "country",

                    color= "count", 

                    hover_name= "country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Viridis,

                    title="Home country of visitors")
#graphing deposit types

sns.countplot(data = df, x = 'deposit_type').set_title('Graph showing types of deposits', fontsize = 20)
#graph repeated guests

sns.countplot(data = df, x = 'is_repeated_guest').set_title('Graph showing whether guest is repeated guest', fontsize = 20)
#graph types of guests

sns.countplot(data = df, x = 'customer_type').set_title('Graph showing type of guest', fontsize = 20)
#graphing prices per month per hotel

#average daily rate = (sumOfAllLodgingTransaction/TotalNumberOfStayingNight)

#average daily rate per person = (ADR/Adults+Children)



# Resizing plot 

plt.figure(figsize=(12,5))



# Calculating average daily rate per person

df['adr_pp'] = df['adr'] / (df['adults'] + df['children']) 

actual_guests = df.loc[df["is_canceled"] == 'not_canceled']

actual_guests['price'] = actual_guests['adr'] * (actual_guests['stays_in_weekend_nights'] + actual_guests['stays_in_week_nights'])

sns.lineplot(data = actual_guests, x = 'arrival_date_month', y = 'price', hue = 'hotel')