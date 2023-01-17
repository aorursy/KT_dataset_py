import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df
df.info()
# Converting reservation_status_date to datetime object

df['reservation_status_date'] = df['reservation_status_date'].astype('datetime64[ns]')
df.isnull().sum()
df.deposit_type.value_counts()
df.is_canceled.value_counts()
df.reservation_status.value_counts()
a = df[df['is_canceled']==0]
b = df[df['is_canceled']==1]
a = a.drop_duplicates()
b = b.drop_duplicates()
a.shape
pd.crosstab(a['stays_in_week_nights'],a['hotel'])
pd.crosstab(a['stays_in_weekend_nights'],a['hotel'])
pd.crosstab(a['hotel'],a['reservation_status'])
pd.crosstab(b['hotel'],b['reservation_status'])
pd.crosstab(a['hotel'],a['deposit_type'])
pd.crosstab(b['hotel'],b['deposit_type'])
a
sns.catplot(x = 'hotel',data=a,kind='count')

plt.xticks(rotation=90)

plt.show()
sns.catplot(x = 'arrival_date_month',data=a,kind='count',hue='hotel')

plt.xticks(rotation=90)

plt.show()
sns.catplot(x = 'customer_type',data=a,kind='count',hue='hotel')

plt.xticks(rotation=90)

plt.show()
sns.distplot(a['lead_time'])

plt.show()
sns.catplot(x = 'market_segment',data=a,kind='count',hue='hotel')

plt.xticks(rotation=90)

plt.show()
sns.catplot(x = 'is_repeated_guest',data=a,kind='count',hue='hotel')

plt.xticks(rotation=90)

plt.show()
sns.catplot(x = 'distribution_channel',data=a,kind='count',hue='hotel')

plt.xticks(rotation=90)

plt.show()



!pip install plotly
country_count = df['country'].value_counts()

country_count_df = pd.DataFrame(country_count)

country_count_df = country_count_df.reset_index()

country_count_df.columns = ['country','booking_count']

country_count_df = country_count_df[country_count_df['booking_count'] > 0]



import plotly.express as px



fig = px.choropleth(country_count_df, locations="country",

                    color="booking_count",

                    hover_name="country")

fig.show()