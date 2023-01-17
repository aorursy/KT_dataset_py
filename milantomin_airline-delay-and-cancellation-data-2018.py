# data processing

import pandas as pd
# Load file (this machine can't handle more)

df = pd.read_csv("/kaggle/input/airline-delay-and-cancellation-data-2009-2018/2018.csv")
# Let's get familiar with the dataset

df.info()
# 7.2M records and 28 columns

# We have (technical) data on airlines, airport, flight number, etc

# Pretty much all other data is time-related (in minutes)
# Set to see all columns

pd.set_option('display.max_columns', None)
df.head()
# Check unique values in OP_CARRIER (airline) column

df.OP_CARRIER.unique()
# Renaming airline codes to company names

# Source: https://en.wikipedia.org/wiki/List_of_airlines_of_the_United_States



df['OP_CARRIER'].replace({

    'UA':'United Airlines',

    'AS':'Alaska Airlines',

    '9E':'Endeavor Air',

    'B6':'JetBlue Airways',

    'EV':'ExpressJet',

    'F9':'Frontier Airlines',

    'G4':'Allegiant Air',

    'HA':'Hawaiian Airlines',

    'MQ':'Envoy Air',

    'NK':'Spirit Airlines',

    'OH':'PSA Airlines',

    'OO':'SkyWest Airlines',

    'VX':'Virgin America',

    'WN':'Southwest Airlines',

    'YV':'Mesa Airline',

    'YX':'Republic Airways',

    'AA':'American Airlines',

    'DL':'Delta Airlines'

},inplace=True)
# Quality check

df.OP_CARRIER.unique()
# Total number of canceled flights

df.CANCELLED.sum()
# Let's explore column CANCELLED

df.CANCELLED.unique()
# From above we see it's binary: 0 or 1, let's see how it looks like

canceled = df[(df['CANCELLED'] > 0)]
canceled.head(3)
# OPTIONAL: Leaving only non-canceled flights

# df = df[(df['CANCELLED'] == 0)]
# Departure delay data (in minutes)

df.DEP_DELAY.head()
# Arrival delay data (in minutes)

df.ARR_DELAY.head()
# To do this analysis right, let's filter all negative numbers in ARR_DELAY column

# Number of delayed flights 

df[df.ARR_DELAY > 0 ].count()
# Filter out non-delayed flights < 0 DEP_DELAY

df = df[(df['ARR_DELAY'] > 0)]
# Minutes to hours 

df['ARR_DELAY'] = df['ARR_DELAY'] / 60



# Minutes to hours 

df['DEP_DELAY'] = df['DEP_DELAY'] / 60
# Down from 7.2 to 2.5 million (relevant) records

df.info()
# Check if FL_DATE is DateTime type

type(df['FL_DATE'])
# Convert string to DateTime

pd.to_datetime(df.FL_DATE)
# Month variable

df['FL_DATE_month'] = pd.to_datetime(df['FL_DATE']).dt.month

# Weekday variable

df['FL_DATE_weekday'] = pd.to_datetime(df['FL_DATE']).dt.weekday_name
import matplotlib.pyplot as plt

%matplotlib inline
# Arrival and departure delays by month of the year

plt.figure(figsize=(25, 12)).subplots_adjust(hspace = 0.5)



plt.subplot(2, 2 ,1)

df.groupby('FL_DATE_month').ARR_DELAY.sum().plot.bar().set_title('ARRIVAL delays by month')

plt.title('ARRIVAL delays by month', fontsize=16)

plt.ylabel('Hours', fontsize=14)

plt.xlabel('Month of the year', fontsize=14)



plt.subplot(2, 2 ,2)

df.groupby('FL_DATE_month').DEP_DELAY.sum().plot.bar()

plt.title('DEPARTURE delays by month', fontsize=16)

plt.ylabel('Hours', fontsize=14)

plt.xlabel('Month of the year', fontsize=14)



plt.show()
# Delays by airlines

plt.figure(figsize=(20, 6))

df.groupby('OP_CARRIER').ARR_DELAY.sum().sort_values(ascending=False).plot.bar()

plt.title('Delays by AIRLINES', fontsize=16)

plt.xlabel('Airline', fontsize=14)

plt.ylabel('Hours', fontsize=14)

plt.show()
# Delays by City

city_by_delay = df.groupby('ORIGIN').ARR_DELAY.sum().sort_values(ascending=False)

plt.figure(figsize=(20, 6))

city_by_delay[:15].plot.bar()

plt.title('Delays by City', fontsize=16)

plt.xlabel('City', fontsize=14)

plt.ylabel('Hours', fontsize=14)

plt.show()