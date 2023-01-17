#imports

import pandas as pd

import numpy as np

from bokeh.charts import Bar, Histogram, TimeSeries, output_file, show

from bokeh.io import output_notebook

from bokeh.models import (

    GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, 

    WheelZoomTool, BoxSelectTool, HoverTool,

    ResizeTool, SaveTool, CrosshairTool

)

from bokeh.charts.attributes import CatAttr, ColorAttr

from bokeh.plotting import figure, output_file, show

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp



#helpers

%matplotlib inline

sns.set_style("whitegrid")

output_notebook()

sigLev = 3

percentLev = 100

alphaLev = .2

binCount = 30

pd.set_option("display.precision",sigLev)
#load in data

trainFrame = pd.read_csv("../input/train.csv")

testFrame = pd.read_csv("../input/test.csv")
print(trainFrame.shape)

print(testFrame.shape)
trainFrame.columns
plt.hist(trainFrame["trip_duration"],bins = binCount)

plt.xlabel("Trip Duration (in seconds)")

plt.ylabel("Count")

plt.title("Distribution of Trip Duration")
trainFrame["logTripDuration"] = np.log(trainFrame["trip_duration"] + 1)

plt.hist(trainFrame["logTripDuration"],bins = binCount)

plt.xlabel("Trip Duration (in Log-Seconds)")

plt.ylabel("Count")

plt.title("Distribution of Trip Duration")
#analyze vendors

vendorCountFrame = trainFrame.groupby("vendor_id",as_index = False)[

                                                                "id"].count()

vendorCountFrame = vendorCountFrame.rename(columns = {"id":"count"})

#then plot

sns.barplot(x = "vendor_id",y = "count",data = vendorCountFrame)

plt.xlabel("Vendor ID")

plt.ylabel("Count")

plt.title("Distribution of Vendors")
plt.hist(trainFrame["passenger_count"],bins = binCount)

plt.xlabel("Number of Passengers")

plt.ylabel("Count")

plt.title("Distribution of Number of Passengers")
#make our datetime objects

trainFrame["pickup_datetime"] = pd.to_datetime(trainFrame["pickup_datetime"])

trainFrame["dropoff_datetime"] = pd.to_datetime(trainFrame["dropoff_datetime"])
#check distribution of days

trainFrame["pickup_dayofyear"] = trainFrame["pickup_datetime"].dt.dayofyear

trainFrame["dropoff_dayofyear"] = trainFrame["dropoff_datetime"].dt.dayofyear

#then plot counts for each

pickupDayCountSeries = trainFrame.groupby("pickup_dayofyear")["id"].count()

dropoffDayCountSeries = trainFrame.groupby("dropoff_dayofyear")["id"].count()

#plot

pickupDayCountSeries.plot(label = "Pickup Day")

dropoffDayCountSeries.plot(label = "Dropoff Day")

plt.legend()

plt.xlabel("Day of year")

plt.ylabel("Count")

plt.title("Taxi Demand over Day of Year")
#check distribution of day of week

trainFrame["pickup_dayofweek"] = trainFrame["pickup_datetime"].dt.dayofweek

trainFrame["dropoff_dayofweek"] = trainFrame["dropoff_datetime"].dt.dayofweek

#then plot counts for each

pickupDayCountSeries = trainFrame.groupby("pickup_dayofweek")["id"].count()

dropoffDayCountSeries = trainFrame.groupby("dropoff_dayofweek")["id"].count()

#plot

pickupDayCountSeries.plot(label = "Pickup Day of Week")

dropoffDayCountSeries.plot(label = "Dropoff Day of Week")

plt.legend()

plt.xlabel("Day of Week")

plt.ylabel("Count")

plt.title("Taxi Demand over Day of Week")
#check distribution of hour

trainFrame["pickup_hour"] = trainFrame["pickup_datetime"].dt.hour

trainFrame["dropoff_hour"] = trainFrame["dropoff_datetime"].dt.hour

#then plot counts for each

pickupDayCountSeries = trainFrame.groupby("pickup_hour")["id"].count()

dropoffDayCountSeries = trainFrame.groupby("dropoff_hour")["id"].count()

#plot

pickupDayCountSeries.plot(label = "Pickup Hour")

dropoffDayCountSeries.plot(label = "Dropoff Hour")

plt.legend()

plt.xlabel("Hour")

plt.ylabel("Count")

plt.title("Taxi Demand over Hours")
plt.scatter(trainFrame["pickup_longitude"],trainFrame["pickup_latitude"],

            alpha = alphaLev,label = "pickup location")

plt.scatter(trainFrame["dropoff_longitude"],trainFrame["dropoff_latitude"],

            alpha = alphaLev,label = "dropoff location")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Longitude and Latitude of Pickups/Dropoffs")

plt.legend()
filteredTrainFrame = trainFrame[(trainFrame["pickup_longitude"] > -100) &

                                (trainFrame["dropoff_longitude"] > -100) &

                                (trainFrame["pickup_latitude"] < 50) &

                                (trainFrame["dropoff_latitude"] < 50)]

#then plot

plt.scatter(filteredTrainFrame["pickup_longitude"],

            filteredTrainFrame["pickup_latitude"],

            alpha = alphaLev,label = "pickup location")

plt.scatter(filteredTrainFrame["dropoff_longitude"],

            filteredTrainFrame["dropoff_latitude"],

            alpha = alphaLev,label = "dropoff location")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Longitude and Latitude of Pickups/Dropoffs")

plt.legend()
filteredTrainFrame = trainFrame[(trainFrame["dropoff_longitude"] > -75) &

                                (trainFrame["dropoff_longitude"] < -72.5) &

                                (trainFrame["dropoff_latitude"] < 42) &

                                (trainFrame["dropoff_latitude"] > 40)]

#then plot

plt.scatter(filteredTrainFrame["dropoff_longitude"],

            filteredTrainFrame["dropoff_latitude"],

            alpha = alphaLev,label = "dropoff location")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Longitude and Latitude of Pickups/Dropoffs")

plt.legend()
storeFlagCountFrame = trainFrame.groupby("store_and_fwd_flag",as_index = False)[

                                            "id"].count()

storeFlagCountFrame = storeFlagCountFrame.rename(columns = {"id":"count"})

#then plot

sns.barplot(x = "store_and_fwd_flag",y = "count",data = storeFlagCountFrame)

plt.xlabel("Store and Forward Flag")

plt.ylabel("Count")

plt.title("Distribution of Store and Foward")
sns.boxplot(x = "vendor_id",y = "logTripDuration",data = trainFrame)

plt.xlabel("Vendor ID")

plt.ylabel("Trip Duration (in Log-Seconds)")

plt.title("Trip Duration on Vendor ID")
plt.scatter(trainFrame["passenger_count"],trainFrame["logTripDuration"],

            alpha = alphaLev)

plt.xlabel("Passenger Count")

plt.ylabel("Trip Duration (in Log-Seconds)")

plt.title("Trip Duration on Passenger Count")
plt.scatter(trainFrame["pickup_dayofweek"],trainFrame["logTripDuration"],

            alpha = alphaLev)

plt.xlabel("Pickup Day of the Week")

plt.ylabel("Trip Duration (in Log-Seconds)")

plt.title("Trip Duration on Day of the Week")
plt.scatter(trainFrame["pickup_hour"],trainFrame["logTripDuration"],

            alpha = alphaLev)

plt.xlabel("Pickup Hour")

plt.ylabel("Trip Duration (in Log-Seconds)")

plt.title("Trip Duration on Pickup Hour")
#get euclidean distance between pickup and dropoff

trainFrame["euclidDist"] = np.sqrt((trainFrame["pickup_latitude"]

                                   -trainFrame["dropoff_latitude"]) ** 2 +

                                   (trainFrame["pickup_longitude"]

                                   -trainFrame["pickup_latitude"]) ** 2)

#then plot relationship

plt.scatter(trainFrame["euclidDist"],trainFrame["logTripDuration"],

            alpha = alphaLev)

plt.xlabel("Euclidean Distance")

plt.ylabel("Trip Duration (in Log-Seconds)")

plt.title("Trip Duration on\nEuclidean Distance of the Trip")