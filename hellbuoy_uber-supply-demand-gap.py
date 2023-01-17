# import all libraries and dependencies for dataframe

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker
# Reading the Uber file



path = '../input/uber-supplydemand-gap/'

file = path + 'Uber Request Data.csv'

uber = pd.read_csv(file)
uber.head()
# Dimensions of df



uber.shape
# Data description



uber.describe()
# Data info



uber.info()
# check if any duplicates record exists



sum(uber.duplicated(subset = "Request id")) == 0
# Calculating the Missing Values % contribution in DF



df_null = uber.isna().mean().round(4)*100



df_null.sort_values(ascending=False)
# Check datatypes of df



uber.dtypes
# Converting the datatype of Request timestamp and Drop timestamp



uber['Request timestamp'] = uber['Request timestamp'].astype(str)

uber['Request timestamp'] = uber['Request timestamp'].str.replace('/','-')

uber['Request timestamp'] = pd.to_datetime(uber['Request timestamp'], dayfirst=True)
uber['Drop timestamp'] = uber['Drop timestamp'].astype(str)

uber['Drop timestamp'] = uber['Drop timestamp'].str.replace('/','-')

uber['Drop timestamp'] = pd.to_datetime(uber['Drop timestamp'], dayfirst=True)
# Extract the hour from the request timestamp



req_hr = uber['Request timestamp'].dt.hour

req_hr.value_counts()

uber['Req hour'] = req_hr
# Extract the day from request timestamp



req_day = uber['Request timestamp'].dt.day

req_day.value_counts()

uber['Req day'] = req_day
# Factor plot of hour and day with respect to Status



sns.factorplot(x = 'Req hour', hue = 'Status', row = 'Req day', data = uber, kind = 'count', size=5, aspect=3)
# Factor plot of hour and day with respect to Pickup Point



sns.factorplot(x = 'Req hour', hue = 'Pickup point', row = 'Req day', data = uber, kind = 'count', size=5, aspect=3)
# Aggregate count plot for all days w.r.t. to Pickup point



sns.factorplot(x = 'Req hour', hue = 'Pickup point', data = uber, kind = 'count', size=5, aspect=3)
# Creating timeslots for various time period of the day



time_hour = [0,5,10,17,22,24]

time_slots =['Early Morning','Morning_Rush','Daytime','Evening_Rush','Late_Night']

uber['Time_slot'] = pd.cut(uber['Req hour'], bins = time_hour, labels = time_slots)
# Visualizing the different time slots wrt status



plt.rcParams['figure.figsize'] = [12,8]

sns.countplot(x = 'Time_slot', hue = 'Status', data = uber)

plt.xlabel("Time Slots",fontweight = 'bold')

plt.ylabel("Number of occurence ",fontweight = 'bold')
# as we can see in the above plot the higest number of cancellations are in the "Morning Rush" time slot

morning_rush = uber[uber['Time_slot'] == 'Morning_Rush']

sns.countplot(x = 'Pickup point', hue = 'Status', data = morning_rush)
# as we can see in the above plot the higest number of no cars available are in the "Evening Rush" time slot

evening_rush = uber[uber['Time_slot'] == 'Evening_Rush']

sns.countplot(x = 'Pickup point', hue = 'Status', data = evening_rush)
# Let's create pie charts instead of a count plots

def pie_chart(dataframe):

    

    labels = dataframe.index.values

    sizes = dataframe['Status'].values

        

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
# percentage breakup of status on the basis of pickup location

# Status of trips @ Morning Rush where pickup point is City

city = uber.loc[(uber["Pickup point"] == "City") & (uber.Time_slot == "Morning_Rush")]

city_count = pd.DataFrame(city.Status.value_counts())

pie_chart(city_count)
# percentage breakup of status on the basis of pickup location

# Status of trips @ Evening Rush where pickup point is City

city = uber.loc[(uber["Pickup point"] == "City") & (uber.Time_slot == "Evening_Rush")]

city_count = pd.DataFrame(city.Status.value_counts())

pie_chart(city_count)
# percentage breakup of status on the basis of pickup location

# Status of trips @ Morning Rush where pickup point is Airport

city = uber.loc[(uber["Pickup point"] == "Airport") & (uber.Time_slot == "Morning_Rush")]

city_count = pd.DataFrame(city.Status.value_counts())

pie_chart(city_count)
# percentage breakup of status on the basis of pickup location

# Status of trips @ Evening Rush where pickup point is Airport

city = uber.loc[(uber["Pickup point"] == "Airport") & (uber.Time_slot == "Evening_Rush")]

city_count = pd.DataFrame(city.Status.value_counts())

pie_chart(city_count)