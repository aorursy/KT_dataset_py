# Here we are importing all the standard packages/libraries necessary to take a look at our data



import json

import numpy as np

import pandas as pd

import folium

from folium import Choropleth

import geopandas

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# It's always a good idea to take a glimpse of the data we're working with.



df = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")

df.head()
# Since I'm only going to be investigating PM10 data, I'll delete the columns containing data for other pollutants. 



unwantedPollutants = {"SO2", "NO2", "O3", "CO", "PM2.5"}

df.drop(unwantedPollutants, axis=1, inplace=True)
# The first thing I wanted to do was create a simple linegraph showing the average PM10 levels for the entire city on a monthly basis.

# In order to achieve this, we'll create a new dataframe called lineplot_df and create a few new columns to help us better organise the data.



lineplot_df = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv", index_col="Measurement date", parse_dates=True)

lineplot_df.drop(unwantedPollutants, axis=1, inplace=True) # Since we created a new dataframe straight from the CSV file rather than copying the main one, we'll delete the other pollutant data again



lineplot_df["Year"] = lineplot_df.index.year

lineplot_df["Month"] = lineplot_df.index.month



# To obtain the monthly average throughout the entire three year period, I'll create a new column that includes both the month and year.

# I'll then group the data by this new column and extract the mean value for each month



lineplot_df["Year and Month"] = lineplot_df[["Year", "Month"]].astype(str).agg("-".join, axis=1)

lineplot_df["PM10avg"] = lineplot_df.groupby(["Year and Month"]).PM10.transform("mean")
# We'll take another quick peek at the data to make sure everything's gone smoothly.



lineplot_df.head()
lineplot_df.shape
# As shown from the above, we've got many unnecessary rows where the PM10 average value has been duplicated. We only really need 36 rows (12 months * 3 years).

# We'll delete all the rows which contain a duplicate value in the "Year and Month" column, as we only need one value for each month.



lineplot_df.drop_duplicates(subset="Year and Month", inplace=True)
# Much better! We now only have the 36 rows we need.



lineplot_df.shape
# Time to plot our linegraph.



ax = plt.figure(figsize=(20,10), constrained_layout=True)

ax.suptitle("Average PM10 for all districts (2017-2019)", fontsize=20, fontweight="bold")

lp = sns.lineplot(x=lineplot_df["Year and Month"], y=lineplot_df["PM10avg"], sort=False)

lp.set_ylabel("Average PM10", fontsize=20)

lp.set_xlabel("Year and Month", fontsize=20)

lp.set_xticklabels(lineplot_df["Year and Month"].values, rotation=40, ha="right")

plt.show()
folium_df = df.copy()



folium_df["PM10addavg"] = folium_df.groupby("Address").PM10.transform("mean")

folium_df.drop_duplicates(subset="Address", inplace=True)
# Making sure it all went smoothly. We should expect to have 25 rows given the 25 unique districts we've got data from.



folium_df.shape
folium_df.head()
# All looks good, time to create the map



m = folium.Map(

    location=[37.5326, 127.0246], # These are the rough coordinates of Seoul

    zoom_start=12

)



for i, row in folium_df.iterrows():

    folium.Marker([row.Latitude, row.Longitude], tooltip="Address: {}.<br> PM10: {}".format(row.Address, row.PM10addavg)).add_to(m)

    

m
# The first thing we need in order to create a choropleth map is data that tells us the boundaries of each district.

# I obtained a GeoJSON file containing such data from a GitHub repository (you can find this at the bottom of the paragraph beneath the map).



with open("../input/seouljson/seoul_municipalities.json", "r") as file:

    district_borders = json.loads(file.read())



test_map = folium.Map(

    location=[37.5326, 127.0246], 

    zoom_start=12)



Choropleth(geo_data=district_borders).add_to(test_map)



for i, row in folium_df.iterrows():

    folium.Marker([row.Latitude, row.Longitude], tooltip="Address: {}.<br> PM10: {}".format(row.Address, row.PM10addavg)).add_to(test_map)



test_map
df.head()
# First we must import the necessary package.



import re



# We know that, conveniently, the data we're looking for all ends with "gu." We can easily create a regular expression that trawls through the entire GeoJSON block of text and extracts only the phrases which satisfy the above condition.



districtsRegex = re.compile(r"""[a-zA-Z.+]+-gu""")

gu_districts = districtsRegex.findall(json.dumps(district_borders))

print(gu_districts)
choro_df = folium_df.copy()

choro_df["name_eng"] = ""



# As we established earlier, all of the district names used in the GeoJSON file are located within the addresses already present in our dataframe.

# We'll now iterate through our new dataframe and allocate each row with its respective GeoJSON district name. 



for i, row in choro_df.iterrows():

    for gu in gu_districts:

        if gu in row.Address:

            choro_df.at[i, "name_eng"] = gu
# Let's take a look at our new dataframe to make sure the names were allocated correctly. The "name_eng" column should now be populated.

# It looks like everything has gone smoothly!



choro_df
# The last thing we must do is set the index of our dataframe as the naming convention used in the GeoJSON file.

# This is because, as you'll soon see below, the "key_on" we'll use when creating the choropleth map will be the "SIG_ENG_NM" property from the GeoJSON data.

# If you remember from earlier, the "SIG_ENG_NM" property is where names such as Jongno-gu, Dongdaemun-gu and Gangnam-gu were found. We're making our index match these names.



choro_df.set_index(choro_df["name_eng"], inplace=True)
# Why not take one last look at our new dataframe just to make sure the new index is all good.



choro_df
m_2 = folium.Map(

    location=[37.5326, 127.0246],

    zoom_start=12

)



Choropleth(geo_data=district_borders,

          data=choro_df["PM10addavg"],

          key_on="feature.properties.SIG_ENG_NM",

          fill_color='YlGnBu').add_to(m_2)



for i, row in choro_df.iterrows():

    folium.Marker([row.Latitude, row.Longitude], tooltip="Station: {}.<br>PM10 Average: {}".format(row.Address, row.PM10addavg)).add_to(m_2)

    



m_2
# First we must create a new dataframe only containing the columns with information about pollutants (we don't need the addresses or latitude/longitudes for this).

# I am going to create two heatmaps, one using the Pearson correlation coefficient and the other using Spearman.

# We deleted the data pertaining to other pollutants from our main dataframe so we'll make a new one taken straight from the CSV file.



corr_df = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv") 

corr_df = corr_df.loc[:, 'SO2':'PM2.5']

correlations = corr_df.corr(method="pearson") # Pearson

sns.heatmap(data=correlations, annot=True)
correlations = corr_df.corr(method="spearman") # Spearman

sns.heatmap(data=correlations, annot=True)