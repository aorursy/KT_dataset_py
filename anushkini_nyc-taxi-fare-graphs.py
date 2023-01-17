# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk("/kaggle"):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc



%matplotlib inline
train_df= pd.read_feather('../input/kernel318ff03a29/nyc_taxi_data_raw.feather')

gc.collect() #used to flush garbage to clear ram
train_df.head()
train_df.describe()
ax = train_df.hist(column='fare_amount',bins = 250,figsize = (25,10))

ax[0][0].set_xlabel("Fare Amount",fontsize = 15)

ax[0][0].set_ylabel("Frequency",fontsize = 15)
#Seems like the frequency of fares above $100 is very low

print("Number of fares greater than 100$: ",len(train_df[train_df['fare_amount'] > 100]))

print("Total rows: ",len(train_df))
ax = train_df[train_df['fare_amount'] <= 100].hist(column='fare_amount',bins = 100,figsize = (25,10))

ax[0][0].set_xlabel("Fare Amount",fontsize = 15)

ax[0][0].set_ylabel("Frequency",fontsize = 15)
#Since the data size is too long we will now take a sample for the rest of the plots

chunksize = 5_000_000

sample_df = train_df[0:chunksize]
#The NYC longitude runs from -74.03 to -73.75 while the latitude runs from 40.63 to 40.85



city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)



sample_df.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='red', 

                s=.02, alpha=.6,figsize=(10,10))



plt.ylim(city_lat_border)

plt.xlim(city_long_border)

plt.xlabel("Dropoff Longitude",fontsize = 15)

plt.ylabel("Dropoff Latitude",fontsize = 15)

plt.imshow(plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true%27'), zorder = 0, extent = (-74.1, -73.7, 40.6, 40.85))
city_long_border = (-74.03, -73.75)

city_lat_border = (40.63, 40.85)



sample_df.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',

                color='blue', 

                s=.02, alpha=.6,figsize=(10,10))



plt.ylim(city_lat_border)

plt.xlim(city_long_border)

plt.xlabel("Pickup Longitude",fontsize = 15)

plt.ylabel("Pickup Latitude",fontsize = 15)

plt.imshow(plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true%27'), zorder = 0, extent = (-74.1, -73.7, 40.6, 40.85))
sample_df.head()
sample_df.describe()
#Let us now check how the passenger_count for each trip varies

#We know that passenger_count varies from 0 to 7. Hence, a historgam of 7 bins will cover this.

ax = sample_df['passenger_count'].value_counts().plot.bar(figsize = (20,10))

ax.set_xlabel("Passenger Count",fontsize = 15)

ax.set_ylabel("Frequency",fontsize = 15)
print("Number of 0 passenger trips: ",len(train_df[train_df['passenger_count'] == 0]))
#Now let us check the correlation beteen passenger_count and fare_amount

ax = sample_df.plot(kind = 'scatter',color = 'green', x = 'passenger_count', y='fare_amount',figsize = (20,10))

ax.set_xlabel("Passenger Count",fontsize = 15)

ax.set_ylabel("Fare Amount",fontsize = 15)
sample_df.groupby("year")['fare_amount'].mean()
sample_df.groupby("year")['fare_amount'].mean().plot.bar(figsize = (20,10))
sample_df.groupby("weekday")['fare_amount'].mean()
sample_df.groupby("weekday")['fare_amount'].mean().plot.bar(figsize = (25,10))
#reference: https://www.kaggle.com/pavanraj159/nyc-taxi-fare-time-series-forecasting



nyc_data  = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows=5_00_000)

nyc_data.head()



coord = ['pickup_longitude','pickup_latitude', 

         'dropoff_longitude', 'dropoff_latitude']



for i in coord :

    nyc_data[i] = nyc_data[i].replace(0,np.nan)

    nyc_data    = nyc_data[nyc_data[i].notnull()]



#Date manipulation

#conver to date format

nyc_data["pickup_datetime"] = nyc_data["pickup_datetime"].str.replace(" UTC","")

nyc_data["pickup_datetime"] = pd.to_datetime(nyc_data["pickup_datetime"],

                                             format="%Y-%m-%d %H:%M:%S")

#extract year

nyc_data["year"]  = pd.DatetimeIndex(nyc_data["pickup_datetime"]).year

#extract month

nyc_data["month"] = pd.DatetimeIndex(nyc_data["pickup_datetime"]).month

nyc_data["month_name"] = nyc_data["month"].map({1:"JAN",2:"FEB",3:"MAR",

                                                4:"APR",5:"MAY",6:"JUN",

                                                7:"JUL",8:"AUG",9:"SEP",

                                                10:"OCT",11:"NOV",12:"DEC"

                                               })

#merge year month

nyc_data["month_year"] = nyc_data["year"].astype(str) + " - " + nyc_data["month_name"]

#extract week day 

nyc_data["week_day"]   = nyc_data["pickup_datetime"].dt.weekday_name

#extract day 

nyc_data["day"]        = nyc_data["pickup_datetime"].dt.day

#extract hour

nyc_data["hour"]        = nyc_data["pickup_datetime"].dt.hour 

nyc_data = nyc_data.sort_values(by = "pickup_datetime",ascending = False)



#Outlier treatment

#drop observations with passengers greater than 6 and equals 0

nyc_data = nyc_data[(nyc_data["passenger_count"] > 0 ) &

                    (nyc_data["passenger_count"] < 7) ]



#drop observations with fareamount  less than 0 and  greater than 99.99% percentile value.

nyc_data = nyc_data[ (nyc_data["fare_amount"] > 0 ) &

                     (nyc_data["fare_amount"]  <  

                      nyc_data["fare_amount"].quantile(.9999))]



#drop outlier observations in data

coords = ['pickup_longitude','pickup_latitude', 

          'dropoff_longitude', 'dropoff_latitude']

for i in coord  : 

    nyc_data = nyc_data[(nyc_data[i]   > nyc_data[i].quantile(.001)) & 

                        (nyc_data[i] < nyc_data[i].quantile(.999))]

    

#create new variable log of fare amount

nyc_data["log_fare_amount"] = np.log(nyc_data["fare_amount"])

    

nyc_data.head()
#Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os #operating system dependent modules of Python

import matplotlib.pyplot as plt #visualization

import seaborn as sns #visualization

%matplotlib inline

import itertools

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization

import warnings

warnings.filterwarnings("ignore")
#reference: https://www.kaggle.com/pavanraj159/nyc-taxi-fare-time-series-forecasting

yrs = [i for i in nyc_data["year"].unique().tolist() if i not in [2015]]



#subset data without year 2015

complete_dat = nyc_data[nyc_data["year"].isin(yrs)]





plt.figure(figsize = (13,15))

for i,j in itertools.zip_longest(yrs,range(len(yrs))) :

    plt.subplot(3,2,j+1)

    trip_counts_mn = complete_dat[complete_dat["year"] == i]["month_name"].value_counts()

    trip_counts_mn = trip_counts_mn.reset_index()

    sns.barplot(trip_counts_mn["index"],trip_counts_mn["month_name"],

                palette = "rainbow",linewidth = 1,

                edgecolor = "k"*complete_dat["month_name"].nunique() 

               )

    plt.title(i,color = "b",fontsize = 12)

    plt.grid(True)

    plt.xlabel("")

    plt.ylabel("trips")
#reference: https://www.kaggle.com/pavanraj159/nyc-taxi-fare-time-series-forecasting

fare_mn = complete_dat.groupby("month_name")["fare_amount"].mean().reset_index()



mnth_ord = ['JAN', 'FEB', 'MAR','APR', 'MAY' , 'JUN',

                'JUL',  'AUG', 'SEP','OCT', 'NOV','DEC']



plt.figure(figsize = (12,7))

sns.barplot("month_name","fare_amount",

            data = fare_mn,order = mnth_ord,

            linewidth =1,edgecolor = "k"*len(mnth_ord)

           )

plt.grid(True)

plt.title("Average fare amount by Month")

plt.xlabel("Month",fontsize = 12)

plt.ylabel("Fare Amount",fontsize = 12)

plt.show()
train_df  = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_00_000)
sns.jointplot(x = train_df.fare_amount, y = train_df.index, data= train_df,size=8, ratio=6, color="#0F336E")
train_df = train_df[(train_df.fare_amount >= 0)]

train_df = train_df[(train_df.fare_amount <= 250)]
sns.jointplot(x = train_df.fare_amount, y = train_df.index, data= train_df,size=8, ratio=6, color="#0F336E",xlim = (0,500))
train_df = train_df[(train_df.passenger_count >= 0)& (train_df.passenger_count <=8)]
train_df.passenger_count
plt.figure(figsize=(20,10))

plt.title("A plot of the average fare amount grouped by passenger count",fontsize = 12)

plt.xlabel("Passenger Count",fontsize = 12)

plt.ylabel("Fare Amount",fontsize = 12)

sns.barplot(x = train_df.passenger_count, y=train_df.fare_amount)
import datetime as dt

train_df['pickup_datetime'] = train_df['pickup_datetime'].str.slice(0, 16)

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

train_df['day'] = train_df['pickup_datetime'].dt.day
plt.figure(figsize = (20,10))

sns.barplot(x = train_df.day, y=train_df.fare_amount)

plt.xlabel("Day of the month",fontsize = 12)

plt.ylabel("Fare Amount",fontsize = 12)

plt.title("Average Fare amount by day of the month", fontsize = 12)