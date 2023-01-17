import numpy as np

import pandas as pd

import datetime



import matplotlib.pyplot as plt

import seaborn as sns

import folium

%matplotlib inline
hotel_bookings = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
hotel_bookings.shape
hotel_bookings.info()
hotel_bookings.describe(include='all')
#Dropping feature "company" as it has 94% NULL.

hotel_bookings = hotel_bookings.drop(axis='1',columns='company')
#Converting certain features to categorical form

categorical_features = ['hotel','is_canceled','arrival_date_week_number','meal','country','market_segment',

                        'distribution_channel','is_repeated_guest','reserved_room_type','assigned_room_type',

                        'deposit_type','agent','customer_type','reservation_status','arrival_date_month']

hotel_bookings[categorical_features] = hotel_bookings[categorical_features].astype('category')



# Converting reservation_status_date to datetime object

hotel_bookings['reservation_status_date'] = hotel_bookings['reservation_status_date'].astype('datetime64[ns]')



# Converting arrival date to datetime object

MonthtoNum = {'January':1, 'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,

             'August':8,'September':9,'October':10,'November':11,'December':12}



hotel_bookings['arrival_date'] = hotel_bookings.apply(lambda x:datetime.date(x['arrival_date_year'],

                                                                             MonthtoNum[x['arrival_date_month']],

                                                                             x['arrival_date_day_of_month']),

                                                      axis = 1)

hotel_bookings['arrival_date'] = hotel_bookings['arrival_date'].astype('datetime64[ns]')



hotel_bookings.info()
# Plot to show outlier in Average Daily Rate

ax = sns.boxplot(x=hotel_bookings['adr'])
hotel_bookings['adr'] = hotel_bookings['adr'].astype('int')
# Deleting a record with ADR greater than 5000

hotel_bookings = hotel_bookings[hotel_bookings['adr'] < 5000]
ax = sns.boxplot(x=hotel_bookings['adr'])
# The function generating the EDA for categorical data



def categorical_eda(df):

    """Given dataframe, generate EDA of categorical data"""

    print("To check: Unique count of non-numeric data")

    print(df.select_dtypes(include=['category']).nunique())

    # Plot count distribution of categorical data

    

    for col in df.select_dtypes(include='category').columns:

        if df[col].nunique() < 20:

            fig = sns.catplot(x=col, kind="count", data=df)

            fig.set_xticklabels(rotation=90)

            plt.show()

        

        

categorical_eda(hotel_bookings)
country_count = hotel_bookings['country'].value_counts()

country_count_df = pd.DataFrame(country_count)

country_count_df = country_count_df.reset_index()

country_count_df.columns = ['country','booking_count']

country_count_df = country_count_df[country_count_df['booking_count'] > 10]



import plotly.express as px



fig = px.choropleth(country_count_df, locations="country",

                    color="booking_count",

                    hover_name="country",

                    color_continuous_scale=px.colors.sequential.RdBu)

fig.show()
reservation_df = hotel_bookings[['hotel','reservation_status']]

reservation_df.groupby(['hotel']).count()
hotel_bookings_1 = hotel_bookings[hotel_bookings['hotel'] == 'City Hotel']

hotel_bookings_1['reservation_status'].value_counts()
hotel_bookings_2 = hotel_bookings[hotel_bookings['hotel'] == 'Resort Hotel']

hotel_bookings_2['reservation_status'].value_counts()
# Percentage of Cancelation

print('Percentage of calculation in City Hotel: ',(32185/79329)*100)

print('Percentage of calculation in Resort Hotel: ',(10831/40060)*100)
import pandas as pd

hotel_bookings = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")