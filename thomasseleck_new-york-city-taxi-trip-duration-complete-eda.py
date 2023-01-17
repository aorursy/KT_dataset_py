# Load main python packages

import math

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 100)

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

import warnings

import seaborn as sns

color = sns.color_palette()



warnings.filterwarnings("ignore")

rcParams['figure.figsize'] = 12, 8

np.random.seed(23)
# Load the data; specify date columns for automatic parsing

trainingSet_df = pd.read_csv("../input/train.csv", parse_dates = ["pickup_datetime", "dropoff_datetime"])

testingSet_df = pd.read_csv("../input/test.csv", parse_dates = ["pickup_datetime"])



# Extract target

target_sr = trainingSet_df["trip_duration"]

trainingSet_df.drop("trip_duration", axis = 1, inplace = True)



# As the column 'dropoff_datetime' is not included in the testing set (to avoid leakage), we remove it from training set

# to avoid overfitting

trainingSet_df.drop("dropoff_datetime", axis = 1, inplace = True)
# Print the features list

trainingSet_df.info()
trainingSet_df.index = trainingSet_df["id"].values

target_sr.index = trainingSet_df["id"].values

trainingSet_df.drop("id", axis = 1, inplace = True)

testingSet_df.index = testingSet_df["id"].values

testingSet_df.drop("id", axis = 1, inplace = True)
sns.distplot(np.log10(target_sr), kde = False)

plt.title("Histogram of the trip duration in seconds")

plt.xlabel("Trip duration (seconds)")

plt.ylabel("Count")
outliersCount = target_sr.loc[(target_sr < 100) | (target_sr > 10000)].shape[0]

print("Outliers count:", outliersCount)

print("Percentage of outliers:", (outliersCount / target_sr.shape[0]) * 100, "%")
trainingSet_df["vendor_id"].value_counts()
plot_df = pd.concat([trainingSet_df["vendor_id"], target_sr], axis = 1)

median = plot_df.groupby("vendor_id")["trip_duration"].median()

sns.boxplot(x = "vendor_id", y = "trip_duration", data = plot_df, order = median.sort_values().index)

plt.title("Distribution of target values depending on the vendor id")
plot_df = pd.concat([trainingSet_df["vendor_id"], target_sr], axis = 1)

plot_df = plot_df.loc[(plot_df["trip_duration"] >= 100) & (plot_df["trip_duration"] <= 10000)] # Remove outliers values

median = plot_df.groupby("vendor_id")["trip_duration"].median()

sns.boxplot(x = "vendor_id", y = "trip_duration", data = plot_df, order = median.sort_values().index)

plt.title("Distribution of target values depending on the vendor id; outliers removed")
trainingSet_df["store_and_fwd_flag"].value_counts()
plot_df = pd.concat([trainingSet_df["store_and_fwd_flag"], target_sr], axis = 1)

plot_df = plot_df.loc[(plot_df["trip_duration"] >= 100) & (plot_df["trip_duration"] <= 10000)] # Remove outliers values

median = plot_df.groupby("store_and_fwd_flag")["trip_duration"].median()

sns.boxplot(x = "store_and_fwd_flag", y = "trip_duration", data = plot_df, order = median.sort_values().index)

plt.title("Distribution of target values depending on the store and forward flag")
trainingSet_df.groupby(["vendor_id", "store_and_fwd_flag"]).size()
sns.distplot(trainingSet_df["passenger_count"], kde = False)

plt.title("Histogram of the number of passengers in each vehicle")

plt.xlabel("Number of passengers")

plt.ylabel("Count")
plot_df = pd.concat([trainingSet_df["passenger_count"], target_sr], axis = 1)

plot_df = plot_df.loc[(plot_df["trip_duration"] >= 100) & (plot_df["trip_duration"] <= 10000)] # Remove outliers values

median = plot_df.groupby("passenger_count")["trip_duration"].median()

sns.boxplot(x = "passenger_count", y = "trip_duration", data = plot_df, order = median.sort_values().index)

plt.title("Distribution of target values depending on the number of passengers inside the car")
trainingSet_df.groupby(["vendor_id", "passenger_count"]).size()
trainingSet_df["pickup_datetime"].hist(bins = 100)
print("Training set time range:", trainingSet_df["pickup_datetime"].min(), "to", trainingSet_df["pickup_datetime"].max())

print("Testing set time range:", testingSet_df["pickup_datetime"].min(), "to", testingSet_df["pickup_datetime"].max())
trainingSet_df["weekday"] = trainingSet_df["pickup_datetime"].dt.weekday

numberOfTrips_sr = trainingSet_df.groupby("weekday").size()

numberOfTrips_sr.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

numberOfTrips_sr.plot.bar()

plt.title("Number of trips for each day of the week")
trainingSet_df["month"] = trainingSet_df["pickup_datetime"].dt.month

numberOfTrips_sr = trainingSet_df.groupby("month").size()

numberOfTrips_sr.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

numberOfTrips_sr.plot.bar()

plt.title("Number of trips for each month")
plot_df = pd.concat([trainingSet_df["pickup_datetime"], target_sr], axis = 1)

plot_df = plot_df.loc[(plot_df["trip_duration"] >= 100) & (plot_df["trip_duration"] <= 10000)] # Remove outliers values

plot_df["weekday"] = plot_df["pickup_datetime"].dt.weekday

numberOfTrips_sr = plot_df.groupby("weekday")["trip_duration"].mean()

numberOfTrips_sr.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

numberOfTrips_sr.plot.bar()

plt.title("Mean duration of trips for each day of the week")
fig = plt.figure()

ax_1 = fig.add_subplot(221)

trainingSet_df["pickup_latitude"].hist(bins = 100, ax = ax_1)

ax_1.set_title("Distribution of pickup latitude")



ax_2 = fig.add_subplot(222)

trainingSet_df["pickup_longitude"].hist(bins = 100, ax = ax_2)

ax_2.set_title("Distribution of pickup longitude")



ax_3 = fig.add_subplot(223)

trainingSet_df["dropoff_latitude"].hist(bins = 100, ax = ax_3)

ax_3.set_title("Distribution of dropoff latitude")



ax_4 = fig.add_subplot(224)

trainingSet_df["dropoff_longitude"].hist(bins = 100, ax = ax_4)

ax_4.set_title("Distribution of dropoff longitude")
trainingSet_df["pickup_latitude"].loc[(trainingSet_df["pickup_latitude"] < 40.5) | (trainingSet_df["pickup_latitude"] > 41)] = 40.8

trainingSet_df["dropoff_latitude"].loc[(trainingSet_df["dropoff_latitude"] < 40.5) | (trainingSet_df["dropoff_latitude"] > 41)] = 40.8

testingSet_df["pickup_latitude"].loc[(testingSet_df["pickup_latitude"] < 40.5) | (testingSet_df["pickup_latitude"] > 41)] = 40.8

testingSet_df["dropoff_latitude"].loc[(testingSet_df["dropoff_latitude"] < 40.5) | (testingSet_df["dropoff_latitude"] > 41)] = 40.8



trainingSet_df["pickup_longitude"].loc[(trainingSet_df["pickup_longitude"] < 40.5) | (trainingSet_df["pickup_longitude"] > 41)] = -74

trainingSet_df["dropoff_longitude"].loc[(trainingSet_df["dropoff_longitude"] < 40.5) | (trainingSet_df["dropoff_longitude"] > 41)] = -74

testingSet_df["pickup_longitude"].loc[(testingSet_df["pickup_longitude"] < 40.5) | (testingSet_df["pickup_longitude"] > 41)] = -74

testingSet_df["dropoff_longitude"].loc[(testingSet_df["dropoff_longitude"] < 40.5) | (testingSet_df["dropoff_longitude"] > 41)] = -74
fig = plt.figure()

ax_1 = fig.add_subplot(221)

trainingSet_df["pickup_latitude"].hist(bins = 100, ax = ax_1)

ax_1.set_title("Distribution of pickup latitude")



ax_2 = fig.add_subplot(222)

trainingSet_df["pickup_longitude"].hist(bins = 100, ax = ax_2)

ax_2.set_title("Distribution of pickup longitude")



ax_3 = fig.add_subplot(223)

trainingSet_df["dropoff_latitude"].hist(bins = 100, ax = ax_3)

ax_3.set_title("Distribution of dropoff latitude")



ax_4 = fig.add_subplot(224)

trainingSet_df["dropoff_longitude"].hist(bins = 100, ax = ax_4)

ax_4.set_title("Distribution of dropoff longitude")
def haversineDistance(x):

    R = 6371e3 # Earth's radius in meters

    origLat = x["pickup_latitude"]

    origLong = x["pickup_longitude"]

    destLat = x["dropoff_latitude"]

    destLong = x["dropoff_longitude"]

    

    phi_1 = math.radians(origLat)

    phi_2 = math.radians(destLat)

    deltaPhi = math.radians(destLat - origLat)

    deltaLambda = math.radians(destLong - origLong)



    a = math.sin(deltaPhi / 2) * math.sin(deltaPhi / 2) + math.cos(phi_1) * math.cos(phi_2) * math.sin(deltaLambda / 2) * math.sin(deltaLambda / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c



trainingSet_df["distance"] = trainingSet_df[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]].apply(haversineDistance, axis = 1)

testingSet_df["distance"] = testingSet_df[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]].apply(haversineDistance, axis = 1)
sns.distplot(trainingSet_df["distance"], kde = False)

plt.title("Histogram of the trip distances")

plt.xlabel("Trip distance (meters)")

plt.ylabel("Count")
plot_df = pd.concat([trainingSet_df, target_sr], axis = 1)

sns.heatmap(plot_df.corr(), annot = True, square = True)