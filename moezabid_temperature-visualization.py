import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
csv_path = "/kaggle/input/temperature-data-set-20052015-michigan-usa/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv"

bin_sized400_path = "/kaggle/input/temperature-data-set-20052015-michigan-usa/BinSize_d400.csv"
# importing Python modules

import matplotlib.pyplot as plt

import mplleaflet

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



def leaflet_plot_stations(binsize, hashid):



    df = pd.read_csv('/kaggle/input/temperature-data-set-20052015-michigan-usa//BinSize_d{}.csv'.format(binsize))



    station_locations_by_hash = df[df['hash'] == hashid]



    lons = station_locations_by_hash['LONGITUDE'].tolist()

    lats = station_locations_by_hash['LATITUDE'].tolist()



    plt.figure(figsize=(8,8))



    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)



    return mplleaflet.display()



leaflet_plot_stations(400, csv_path)
# Loading the data

temperature_data = pd.read_csv(csv_path)
# Getting data during the period of 2005-2014

data = temperature_data[(temperature_data["Date"] >= "2005-01-01") & (temperature_data["Date"] <= "2014-12-31")]

# Getting data during the year 2015

data_2015 = temperature_data[(temperature_data["Date"] >= "2015-01-01") & (temperature_data["Date"] <= "2015-12-31")]
# Removing Leap days

data = data[~data.Date.str.endswith('02-29')].copy()
# Sorting the data by Date

data = data.sort_values("Date")
data.head()
data_2015.head()
# Converting the "Date" column to datetime

data["Date"] = list(map(pd.to_datetime, data["Date"]))
# Diving the data into two dataframes for high and low

high = data[data["Element"] == "TMAX"]

low = data[data["Element"] == "TMIN"]
# Getting record high and low temperature values for each day of the year during the period of 2004-2015

record_high = high.copy()

record_high['dayofyear'] = record_high['Date'].map(lambda x: x.replace(year=2015).dayofyear)

record_high = record_high.groupby("dayofyear").max()



record_low = low.copy()

record_low['dayofyear'] = record_low['Date'].map(lambda x: x.replace(year=2015).dayofyear)

record_low = record_low.groupby("dayofyear").min()
# Sorting values by Date

data_2015 = data_2015.sort_values("Date")

# Converting dates to datetime type

data_2015["Date"] = list(map(pd.to_datetime, data_2015["Date"]))

# Diving the data into two dataframes for high and low

high_2015 = data_2015[data_2015["Element"] == "TMAX"]

low_2015 = data_2015[data_2015["Element"] == "TMIN"]



# Getting record high and low temperature values for each day of the year 2015

record_high_2015 = high_2015.copy()

record_high_2015["dayofyear"] = record_high_2015["Date"].dt.dayofyear

record_high_2015 = record_high_2015.groupby("dayofyear").max()



record_low_2015 = low_2015.copy()

record_low_2015["dayofyear"] = record_low_2015["Date"].dt.dayofyear

record_low_2015 = record_low_2015.groupby("dayofyear").min()
# Reseting dataframes indexes

record_low = record_low.reset_index()

record_high = record_high.reset_index()

record_low_2015 = record_low_2015.reset_index()

record_high_2015 = record_high_2015.reset_index()
# Getting indexes of highs and lows that were broken

broken_lows = (record_low_2015[record_low_2015["Data_Value"] < record_low['Data_Value']]).index.tolist()

broken_highs = (record_high_2015[record_high_2015['Data_Value'] > record_high['Data_Value']]).index.tolist()
plt.figure(figsize=(20,7))

plt.plot(record_high["Data_Value"], c="r", alpha=0.8, label = 'Record High 2005-2014')

plt.plot(record_low["Data_Value"], c="b", alpha=0.8, label = 'Record Low 2005-2014')

plt.scatter(broken_lows, record_low_2015['Data_Value'].iloc[broken_lows], s=20, c = 'black', label = 'Record Low broken in 2015')

plt.scatter(broken_highs, record_high_2015['Data_Value'].iloc[broken_highs], s=20, c = 'b', alpha=0.8, label = 'Record High broken in 2015')

plt.legend()

plt.title("2015's temperature breaking points against 2005-2014 in Ann Arbor, Michigan, US")

plt.fill_between(range(len(record_low)),

                       record_low["Data_Value"], record_high["Data_Value"], 

                       facecolor='pink', 

                       alpha=0.11);

# Aligning plot

plt.gca().axis([-1, 365, -400, 450])



# Hiding plot spines

plt.gca().spines['top'].set_visible(False)

plt.gca().spines['right'].set_visible(False)



# Changing Vertical and Horizontal Ticks labels

month_ticks = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

divs = [i+15 for i in month_ticks]

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.xticks(divs, month_names)

temp = [str(tick/10)+str(' °C') for tick in plt.gca().get_yticks()]

plt.gca().set_yticklabels(temp);

plt.savefig('Temp_Plot.png');