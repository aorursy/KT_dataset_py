import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

import folium

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv", index_col = "id")

# read the input data from file with the "id" as index
df.head(n = 5)

# display top 5 rows of the data
df["pickup_datetime"] = pd.DatetimeIndex(df["pickup_datetime"], dtype = pd.DatetimeIndex)

# convert to datetime object
holidays_df = pd.read_csv("../input/nycholidays/NYC_holidays.csv")

# read the holidays data

holidays_df["Date"] = pd.DatetimeIndex(holidays_df["Date"])

# convert to datetime object

HOLIDAYS = holidays_df["Date"].apply(lambda x : x.date()).values

# get the holiday/event dates
def get_datetime_details(pickup_datetime, df):

    """

    Get more details related date and time

    which will be useful for our analysis.

    """

    

    df = df.assign(hour = pickup_datetime.dt.hour)

    # add hours column to the dataframe

    df = df.assign(minute = pickup_datetime.dt.minute)

    # add minute column to the dataframe

    df = df.assign(second = pickup_datetime.dt.second)

    # add second column to the dataframe

    df = df.assign(date = pickup_datetime.dt.date)

    # add date column to the dataframe

    df = df.assign(day = pickup_datetime.dt.dayofweek)

    # add the day column to the dataframe

    df = df.assign(weekend_or_not = df["day"].apply(lambda x: x >= 5))

    df["weekend_or_not"] = df["weekend_or_not"].astype(int)

    # check for weekend

    df = df.assign(holiday_or_not = df["date"].apply(lambda x: x in HOLIDAYS))

    df["holiday_or_not"] = df["holiday_or_not"].astype(int)

    # check if its a holiday

    return df
df = get_datetime_details(df["pickup_datetime"], df)

# add details
df.head()
route_df1 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv",

                      index_col = "id")

route_df2 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv",

                      index_col = "id")

route_df = pd.concat([route_df1, route_df2])

# read the routes data which will aid our analysis
route_df.head()
maneuver_count_df = pd.read_csv("../input/nycholidays/various_maneuver_counts.csv", index_col="id")

# get the various maneuver counts
def add_route_details(df, route_df):

    """

    This function helps to add various route details

    to the main dataframe. This will assist us in 

    further analysis.

    """

    

    df = df.assign(distance = route_df["total_distance"])

    # add the ride distance as a feature

    df = df.assign(best_travel_time = route_df["total_travel_time"])

    # get the best travel time

    return df
df = add_route_details(df, route_df)

# add the distance and best time possible
df = df.join(maneuver_count_df)

# get the maneuver counts as well
weather_df = pd.read_csv("../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv")

# load the weather data

weather_df.head()
weather_df.replace(to_replace = "T", value = 0.0, inplace = True)
weather_df["snow fall"] = weather_df["snow fall"].apply(lambda x : pd.to_numeric(x))

weather_df["snow depth"] = weather_df["snow depth"].apply(lambda x : pd.to_numeric(x))

weather_df["precipitation"] = weather_df["precipitation"].apply(lambda x : pd.to_numeric(x))

# make the column datatypes as float
def convert_to_date_format(row):

    """

    Convert given string to datetime format

    """

    

    row_lst = row.split("-")

    return datetime.date(year = int(row_lst[2]), month = int(row_lst[1]), day = int(row_lst[0]))



weather_df["date"] = weather_df["date"].apply(convert_to_date_format)
df = pd.merge(df, weather_df, on = "date", right_index=True)

# add the weather data

weather_df["date"] = pd.DatetimeIndex(weather_df["date"])
df.head()
df.dropna(inplace = True, how = "any")

# drop rows with empty values
trip_dur_desc = df["trip_duration"].describe()

# get the description about the data
trip_dur_desc
df = df[(df["trip_duration"] > 15) & (df["trip_duration"] < 18000)]

# filter out data
mini_log = np.log10(trip_dur_desc["min"])

maxi_log = np.log10(trip_dur_desc["max"])

n_bins = int(np.log2(trip_dur_desc["count"])) + 1

log_bins = np.logspace(mini_log, maxi_log, n_bins)
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5), dpi = 96)

ax1.boxplot(x = df["trip_duration"], showfliers = True, showmeans = True)

ax1.set_yscale("log")

ax1.set_title("Box plot of trip duration (seconds) in log scale")

ax2.hist(x = df["trip_duration"], bins = log_bins)

ax2.set_xscale("log")

ax2.set_xlabel("trip duration")

ax2.set_ylabel("count")

ax2.set_title("Histogram of trip duration")
trip_dur_desc
sns.set()

g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)

g.map(sns.boxplot, "trip_duration", orient = "v")

sns.set(font_scale = 1.5)

plt.yscale("log")

plt.suptitle("Box plot of trip duration (seconds) in log scale across vendors")

plt.subplots_adjust(top = 0.85)
sns.set(font_scale = 1.5)

g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)

g.map(sns.distplot, "trip_duration", bins = log_bins, kde = False)

plt.suptitle("Histogram of trip duration (seconds) across vendors")

plt.xscale("log")

plt.subplots_adjust(top = 0.85)
sample_df = df.sample(2000)

# get 10% of data
loc = (sample_df["pickup_latitude"][0], sample_df["pickup_longitude"][0])

# first location
def add_pickup_loc_to_map(row):

    loc = (row["pickup_latitude"], row["pickup_longitude"])

    if row.vendor_id == 1:

        col = "orange"

    else:

        col = "blue"

    folium.CircleMarker(location = loc, radius = 2.5, color = col, fill_opacity = 0.5,

                       fill_color = col, weight = 1,

                       popup = "Trip ID:{}\nVendor ID:{}".format(row.name,

                       row.vendor_id)).add_to(pickup_cluster)

    

def add_dropoff_loc_to_map(row):

    loc = (row["dropoff_latitude"], row["dropoff_longitude"])

    if row.vendor_id == 1:

        col = "orange"

    else:

        col = "blue"

    folium.CircleMarker(location = loc, radius = 2.5, color = col, fill_opacity = 0.5,

                       fill_color = col, popup = "Trip ID:{}\nVendor ID:{}".format(row.name,

                                                                                  row.vendor_id),

                        weight = 1).add_to(dropoff_cluster)
nyc_pickup_map = folium.Map(location = loc, tiles = "CartoDB positron", zoom_start = 11)

pickup_cluster = folium.MarkerCluster().add_to(nyc_pickup_map)
sample_df.apply(add_pickup_loc_to_map, axis = 1)

nyc_pickup_map
nyc_dropoff_map = folium.Map(location = loc, tiles = "CartoDB positron", zoom_start = 11)

dropoff_cluster = folium.MarkerCluster().add_to(nyc_dropoff_map)
sample_df.apply(add_dropoff_loc_to_map, axis = 1)

nyc_dropoff_map
sns.set()

g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)

g.map(sns.boxplot, "passenger_count", orient = "v")

sns.set(font_scale = 1.5)

plt.suptitle("Box plot of passenger count across vendors")

plt.subplots_adjust(top = 0.85)
sns.set()

g = sns.FacetGrid(df, col = "vendor_id", size = 6, aspect = 1)

g.map(sns.distplot, "passenger_count", bins = 10, kde = False)

sns.set(font_scale = 1.5)

plt.suptitle("Histogram of passenger count across vendors")

plt.subplots_adjust(top = 0.85)
sns.set_color_codes()

g = sns.FacetGrid(data = df, col = "vendor_id", size = 6)

g.map(sns.boxplot, "passenger_count", "trip_duration", palette = "Blues")

plt.yscale("log")

plt.suptitle("Boxplot of trip duration with various passenger counts for each vendor")

plt.subplots_adjust(top = 0.85)
pass_count_zero_df = df.query(expr = "passenger_count == 0")

# data with passenger count zero

pass_count_zero_df.sort_values(by = "trip_duration", inplace = True)
print("Data points with passenger count zero is {}".format(len(pass_count_zero_df)))
plt.scatter(x = range(0, len(pass_count_zero_df)), y = pass_count_zero_df["trip_duration"])

plt.yscale("log")

plt.title("trip duration when the passenger count is zero")

plt.ylabel("trip duration")

plt.xlabel("trip number")
df = df.query("passenger_count != 0")

# remove rows with passenger count is zero
sns.set_color_codes()

g = sns.FacetGrid(data = df, col = "vendor_id", size = 6)

g.map(sns.boxplot, "passenger_count", "trip_duration", palette = "Blues")

plt.yscale("log")

plt.suptitle("Boxplot of trip duration with various passenger counts for each vendor")

plt.subplots_adjust(top = 0.85)
corr = np.corrcoef(x = df["trip_duration"], y = df["passenger_count"])

print("Correlation between trip duration and passenger count: {:.4f}".format(corr[0][1]))
sns.set_color_codes()

g = sns.FacetGrid(data = df, col = "vendor_id", size = 6)

g.map(sns.boxplot, "passenger_count", "distance", palette = "Blues")

plt.yscale("log")

plt.suptitle("Boxplot of distance with various passenger counts for each vendor")

plt.subplots_adjust(top = 0.85)
corr = np.corrcoef(x = df["distance"], y = df["passenger_count"])

print("Correlation between distance and passenger count: {:.4f}".format(corr[0][1]))
start_date = min(df["date"])

end_date = max(df["date"])
print("The start and end date are as follows")

print(start_date.ctime())

print(end_date.ctime())
date_df = df[["vendor_id", "date"]]

date_df = date_df.assign(trip_count_per_day = df.index)
date_df_group = date_df.groupby(["vendor_id", "date"])
date_count_df = date_df_group.count()
mon_fri_dates = weather_df[(weather_df["date"].dt.dayofweek == 0) | (weather_df["date"].dt.dayofweek == 4)]["date"]
ax = date_count_df.unstack(0).plot(figsize = (20, 5))

ax.legend(["vendor 1", "vendor 2"])

plt.title("Trip counts per day across vendors")

plt.ylabel("Count")

for i in range(len(mon_fri_dates)):

    if i % 2 == 0:

        color = "red"

    else:

        color = "violet"

    ax.axvline(x = mon_fri_dates.iloc[i], color = color, alpha = 0.9, linestyle = "--")
weather_df_train = weather_df[weather_df["date"] <= end_date]

# filter out for dates that are in training
weather_df_train[["precipitation", "snow fall", "snow depth"]].plot(figsize = (12, 5))

plt.title("Weather across all days in training set")

plt.axvline()
day_df = df[["day", "vendor_id"]]

day_df = day_df.assign(trip_id = df.index)

day_df_group = day_df.groupby(["vendor_id", "day"])

day_count_df = day_df_group.count()
hour_df = df[["hour", "vendor_id"]]

hour_df = hour_df.assign(trip_id = df.index)

# get the hour data
hour_df_group = hour_df.groupby(["vendor_id", "hour"])

hour_count_df = hour_df_group.count()
ax = hour_count_df.unstack(0).plot(figsize = (14, 5))

ax.legend(["vendor 1", "vendor 2"])

plt.title("Trip counts per hours across vendors")

plt.ylabel("Count")
hour_day_group_df = df[["hour", "day", "trip_duration"]].groupby(["hour", "day"]).mean()
ax1 = hour_day_group_df.unstack(1).plot(figsize = (18, 8))

ax1.legend(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

plt.title("Trip duration across hours for all days")

plt.ylabel("Trip duration (seconds)")
holiday_data_df = df.query("holiday_or_not == 1")

# trips done on holidays alone

holiday_not_df = df.query("holiday_or_not != 1")

# trips that are done on non-holiday days
holiday_data_df_group = holiday_data_df[["date", "vendor_id"]].groupby("date").count()

holiday_not_df_group = holiday_not_df[["date", "vendor_id"]].groupby("date").count()
ax1 = holiday_data_df_group.plot(figsize = (14, 5))

holiday_not_df_group.plot(ax = ax1)

ax1.legend(["holiday/event", "not holiday/event"])

plt.title("Number of trips done per day based on holiday or not a holiday")

plt.ylabel("count")
df.columns
columns_to_remove = ["dropoff_datetime", "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag",

                    "minute", "second", "date", "distance", "best_travel_time", "maximum temperature",

                    "minimum temperature", "average temperature"]