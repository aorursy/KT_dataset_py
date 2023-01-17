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
# Importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
data = pd.read_csv('/kaggle/input/hotelbookings/hotel_bookings.csv')

data.head()
data.shape
data.isnull().sum()
data = data.drop(columns = ['agent', 'company'])
data = data.dropna(axis = 0)
# Enlarging the pie chart

plt.rcParams['figure.figsize'] = 8,8



# Indexing labels. tolist() will convert the index to list for easy manipulation

labels = data['hotel'].value_counts().index.tolist()



# Convert value counts to list

sizes = data['hotel'].value_counts().tolist()



# As the name suggest, explode will determine how much each section is separated from each other 

explode = (0, 0.1)



# Determine colour of pie chart

colors = ['lightskyblue','yellow']



# Putting them together. Sizes with the count, explode with the magnitude of separation between pies, colors with the colors, 

# autopct enables you to display the percent value using Python string formatting. .1f% will round off to the tenth place.

# startangle will allow the percentage to rotate counter-clockwise. Lets say we have 4 portions: 10%, 30%, 20% and 40%. The pie will rotate from smallest to the biggest (counter clockwise). 10% -> 20% -> 30% -> 40%

# We have only 2 sections so anglestart does not matter

# textprops will adjust the size of text

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',startangle=90, textprops={'fontsize': 14})
data['adults'].groupby(data['hotel']).describe()
data['children'].groupby(data['hotel']).describe()
data['is_canceled'] = data.is_canceled.replace([1,0], ['canceled', 'not canceled'])

canceled_data = data['is_canceled']

sns.countplot(canceled_data)
lst1 = ['is_canceled', 'hotel']

type_of_hotel_canceled = data[lst1]

canceled_hotel = type_of_hotel_canceled[type_of_hotel_canceled['is_canceled'] == 'canceled'].groupby(['hotel']).size().reset_index(name = 'count')

sns.barplot(data = canceled_hotel, x = 'hotel', y = 'count').set_title('Graph showing cancellation rates in city and resort hotel')
# Look into arrival year (arrival_date_year)

lst3 = ['hotel', 'arrival_date_year', 'arrival_date_month','arrival_date_day_of_month' ]

period_arrival = data[lst3]



# Countplot is useful for working with categorical data. It creates a count for you automatically, which means you don't need to write lines of code to calculate. 

sns.countplot(data = period_arrival, x = 'arrival_date_year', hue = 'hotel') #In seaborn, the hue parameter determines which column in the data frame should be used for colour encoding.
plt.figure(figsize=(20,5)) # adjust the size of the plot



# Countplot is useful for working with categorical data. It creates a count for you automatically, which means you don't need to write lines of code to calculate. 

sns.countplot(data = period_arrival, x = 'arrival_date_month', hue = 'hotel', order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December']).set_title('Graph showing number of arrival per month',fontsize=20)

plt.xlabel('Month') # Creating label for xaxis

plt.ylabel('Count') # Creating label for yaxis
plt.figure(figsize=(15,5))



sns.countplot(data = period_arrival, x = 'arrival_date_day_of_month', hue = 'hotel').set_title('Graph showing number of arrival per day', fontsize = 20)
sns.countplot(data = data, x = 'stays_in_weekend_nights').set_title('Number of stays on weekend nights', fontsize = 20)
sns.countplot(data = data, x = 'stays_in_week_nights' ).set_title('Number of stays on weekday night' , fontsize = 20)
sns.countplot(data = data, x = 'adults', hue = 'hotel').set_title("Number of adults", fontsize = 20)
sns.countplot(data = data, x = 'children', hue = 'hotel').set_title("Number of children", fontsize = 20)
sns.countplot(data = data, x = 'babies', hue = 'hotel').set_title("Number of babies", fontsize = 20)
country_visitors = data[data['is_canceled'] == 'not_canceled'].groupby(['country']).size().reset_index(name = 'count')



# We will be using Plotly.express to plot a choropleth map. Big fan of Plotly here!

import plotly.express as px



px.choropleth(country_visitors,

                    locations = "country",

                    color= "count", 

                    hover_name= "country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title="Home country of visitors")
plt.figure(figsize=(12,5))

sns.countplot(data = data, x = 'market_segment').set_title('Types of market segment', fontsize = 20)
plt.figure(figsize=(12,5))

sns.countplot(data = data, x = 'distribution_channel').set_title('Types of distribution channel', fontsize = 20)
sns.countplot(data = data, x = 'deposit_type').set_title('Graph showing types of deposits', fontsize = 20)
sns.countplot(data = data, x = 'is_repeated_guest').set_title('Graph showing whether guest is repeated guest', fontsize = 20)
sns.countplot(data = data, x = 'customer_type').set_title('Graph showing type of guest', fontsize = 20)
# Resizing plot 

plt.figure(figsize=(12,5))



# Calculating average daily rate per person

data['adr_pp'] = data['adr'] / (data['adults'] + data['children']) 

actual_guests = data.loc[data["is_canceled"] == 'not_canceled']

actual_guests['price'] = actual_guests['adr'] * (actual_guests['stays_in_weekend_nights'] + actual_guests['stays_in_week_nights'])

sns.lineplot(data = actual_guests, x = 'arrival_date_month', y = 'price', hue = 'hotel')