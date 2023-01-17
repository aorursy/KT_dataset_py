# Import libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import seaborn as sns



# Configure matplotlib to deal with dark themes

plt.style.use("default")



# Set pandas to show all Dataframes columns and rows.

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# Create a dictionary of column descriptions. It will be util when plotting those features.

columns_descriptions = {

    "wsid": "Weather station id",

    "wsnm": "Name station (usually city location or nickname)",

    "elvt": "Elevation",

    "lat": "Latitude",

    "lon": "Longitude",

    "inme": "Station number (INMET number) for the location",

    "city": "City",

    "prov": "State (Province)",

    "mdct": "Observation Datetime (complete date: date + time)",

    "date": "Date of observation",

    "yr": "The year (2000-2016)",

    "mo": "The month (0-12)",

    "da": "The day (0-31)",

    "hr": "The hour (0-23)",

    "prcp": "Amount of precipitation in millimetres (last hour)",

    "stp": "Air pressure for the hour in hPa to tenths (instant)",

    "smax": "Maximum air pressure for the last hour in hPa to tenths",

    "smin": "Minimum air pressure for the last hour in hPa to tenths",

    "gbrd": "Solar radiation KJ/m2",

    "temp": "Air temperature (instant) in celsius degrees",

    "dewp": "Dew point temperature (instant) in celsius degrees",

    "tmax": "Maximum temperature for the last hour in celsius degrees",

    "dmax": "Maximum dew point temperature for the last hour in celsius degrees",

    "tmin": "Minimum temperature for the last hour in celsius degrees",

    "dmin": "Minimum dew point temperature for the last hour in celsius degrees",

    "hmdy": "Relative humid in % (instant)",

    "hmax": "Maximum relative humid temperature for the last hour in %",

    "hmin": "Minimum relative humid temperature for the last hour in %",

    "wdsp": "Wind speed in metres per second",

    "wdct": "Wind direction in radius degrees (0-360)",

    "gust": "Wind gust in metres per second",

}



np.random.seed(13)

# Create a dictionary of colors.

columns_colors = {

    color: (np.random.random(3).tolist()) for color in columns_descriptions.keys()

}
# Read the dataset. We also parse the mdct column to Datetime objects.

df = pd.read_csv("/kaggle/input/hourly-weather-surface-brazil-southeast-region/sudeste.csv", parse_dates=["mdct"])



# Show some info.

df.info()
df.drop(["wsnm", "inme", "city", "prov", "date"], axis=1, inplace=True)



# Set the index to wsid and mdct

df = df.set_index("mdct")



# Get data from only last six years

df["2010": "2016"]



# Show first entries

df.head()
na_values = df.isna().sum()

na_values
# Columns with missing values

na_columns = df.columns[na_values != 0].tolist()



# Check the missing values by station

df[na_columns].isna().groupby(df.wsid).sum()
# Select data from station 317.

st_317 = df[df.wsid == 317].copy()

st_317.head()
# Utility function to plot the station data.

def plot_station_data(data, col_name, ax):

    title = columns_descriptions[col_name]

    color = columns_colors[col_name]

    ax.plot(data, color=color)

    ax.set(title=title,

       ylabel=col_name)
# Plot columns with missing values from year 2015.

na_columns = st_317.columns[st_317.isna().sum() != 0].tolist()

fig, axs = plt.subplots(nrows=len(na_columns), ncols=1, figsize=(20, 20))



for ax, col_name in zip(axs, na_columns):

    plot_station_data(st_317[col_name]["2015"], col_name, ax)



fig.tight_layout(pad=1.0)

fig.show()
# Plot gbrd data from one week.

fig, ax = plt.subplots(figsize=(20, 5))

plot_station_data(st_317.gbrd["2015-09-01": "2015-09-07"], "gbrd", ax)

fig.show()
zeros_columns = ["prcp", "gbrd"]  # zeros

interpolate_columns = ["wdsp", "gust", "temp", "dewp", "tmax", "dmax", "tmin", "dmin", "hmax", "hmin"]  # interpolations



# Fill missing values with with zeros

st_317[zeros_columns] = st_317[zeros_columns].fillna(0)

# Linearlly interpolate missing values.

st_317[interpolate_columns] = st_317[interpolate_columns].interpolate(method="linear")



# Check for missing values again.

st_317.isna().sum()
st_317_original_shape = st_317.shape
# Select rows with all feature columns equal to zero.

feature_columns = [

    "prcp",

    "stp",

    "smax",

    "smin",

    "gbrd",

    "temp",

    "dewp",

    "tmax",

    "dmax",

    "tmin",

    "dmin",

    "hmdy",

    "hmax",

    "hmin",

    "wdsp",

    "wdct",

    "gust",

]

st_317 = st_317[(st_317[feature_columns] != 0).any(axis=1)]

st_317.head()
st_317_original_shape, st_317.shape
st_317.info()
st_317 = st_317.drop(["dewp", "tmax", "dmax", "tmin", "dmin"], axis=1)
corr_matrix = st_317.corr()
fix, ax = plt.subplots(figsize=(15, 10))

ax = sns.heatmap(corr_matrix,

                annot=True,

                linewidths=0.5,

                fmt=".2f",

                cmap="YlGnBu")

ax.set(title="Correlation matrix");
# Split the data.

np.random.seed(13)



X_train, X_test, y_train, y_test = train_test_split(st_317.drop("temp", axis=1), st_317["temp"], test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Define the model

model = RandomForestRegressor(n_estimators=100)



# Train

model.fit(X_train, y_train)
# Compute the score of the model on the train data.

model.score(X_train, y_train)
# Compute the score of the model on the test data.

model.score(X_test, y_test)