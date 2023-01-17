# Importing the libraries



import pandas as pd

import datetime



# for visualizations



import folium

from folium import plugins

from folium.plugins import HeatMap

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
# Importing the dataset. Checking the names of columns.



df = pd.read_csv('/kaggle/input/crime-in-baltimore/BPD_Part_1_Victim_Based_Crime_Data.csv')

df.info()
# Renaming long and uninformative "Description" to laconical "Crime"



df.rename(columns={'Description':'Crime'}, inplace=True)

df['Crime'] = df['Crime'].str.capitalize()
# How many accidents we have



df.shape
# To return median value per day



df['CrimeDate'].value_counts().median()
# Take a dates with a minimum of crimes



df['CrimeDate'].value_counts().tail(3)
# Take a dates with a maximum of crimes



df['CrimeDate'].value_counts().head(3)
# Analysing the day of Freddy Gray funeral



df_fgray = df['CrimeDate'] == '04/27/2015'

df.loc[df_fgray]['Crime'].value_counts().plot.bar(figsize=(11, 6))
# Median value again



df['CrimeDate'].value_counts().median()
# Building a bar plot with full stats of Baltimore crimes



crime_num = df['Crime'].value_counts().plot.bar(figsize=(11, 6))
# How many crimes and percentage are in called time period.



crime_num = df['Crime'].value_counts()

crime_pct = df['Crime'].value_counts(1) * 100

pd.DataFrame({'Crimes': crime_num, 'Percent' : crime_pct}).round(2)
# Return pie plot with cuts of crime



plt.rcParams.update({'font.size': 12})

crime_num = df['Crime'].value_counts().head(15).plot.pie(radius=4, autopct='%1.1f%%', textprops=dict(color="black"))
# Taking back the font for the bar plot



plt.rcParams.update({'font.size': 18})



# Date format correction



df['CrimeDate'] = pd.to_datetime(df['CrimeDate'], format='%m/%d/%Y')



# Define day of week and column creation



df['DayOfWeek'] = df['CrimeDate'].dt.day_name()



# What part of the week is most dangerous?



df['DayOfWeek'].value_counts().plot.bar(figsize=(11, 6))
# Creating dataframe with homicides and returning on the bar plot to find out which neighborhoods are most dangerous by statistics of murders



hc = df['Crime'] == 'Homicide'

df.loc[hc]['Neighborhood'].value_counts().head(35).plot.bar(figsize=(26, 6))
# Creating dataframe with shootings and returning on the bar plot to find out which neighborhoods are most dangerous by statistics of shooting



sh = df['Crime'] == 'Shooting'

df.loc[sh]['Neighborhood'].value_counts().head(35).plot.bar(figsize=(26, 6))
# Creating dataframe with aggravated assaults and returning in the bar plot to find out which neighborhoods are most dangerous by statistics of aggravated assaults



ass = df['Crime'] == 'Agg. assault'

df.loc[ass]['Neighborhood'].value_counts().head(35).plot.bar(figsize=(26, 6))
hc = df['Crime'] == 'Homicide'

df.loc[hc]['CrimeTime'].value_counts()
hc = df['Crime'] == 'Shooting'

df.loc[hc]['CrimeTime'].value_counts().head(8)
sh = df['Crime'] == 'Homicide'

df.loc[sh]['DayOfWeek'].value_counts().head(7)
sh = df['Crime'] == 'Agg. assault'

df.loc[sh]['DayOfWeek'].value_counts().sort_index().plot.bar(figsize=(12, 6))
# Dropping crimes with NaN numbers of neccessaried data



df = df[pd.notnull(df['Latitude'])]
# Dropping crimes with NaN numbers of neccessaried data



df = df[pd.notnull(df['Longitude'])]
# Taking a look how many  accidents left. Enough.



df.shape
# Limitation of the area which we need



m = folium.Map(location=[39.3121, -76.6198], zoom_start=13)

m
# for index, row in df.iterrows():

#    folium.CircleMarker([row['Latitude'], row['Longitude']],

#                        radius=15,

#                        popup=row['Crime'],

#                        fill_color="#3db7e4", # divvy color

#                       ).add_to(m)
# Ensure you're handing it floats

df['Latitude'] = df['Latitude'].astype(float)

df['Longitude'] = df['Longitude'].astype(float)



# Filter the DF for rows, then columns, then remove NaNs

# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster

heat_df = df[df['Crime']=='Homicide'] # Reducing data size so it runs faster

# heat_df = df[['Latitude', 'Longitude']]

# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])



# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(m)



# Display the map

m
# Ensure you're handing it floats

df['Latitude'] = df['Latitude'].astype(float)

df['Longitude'] = df['Longitude'].astype(float)



# Filter the DF for rows, then columns, then remove NaNs

# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster

heat_df = df[df['Crime']=='Shooting'] # Reducing data size so it runs faster

# heat_df = df[['Latitude', 'Longitude']]

# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])



# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(m)



# Display the map

m
# Ensure you're handing it floats

df['Latitude'] = df['Latitude'].astype(float)

df['Longitude'] = df['Longitude'].astype(float)



# Filter the DF for rows, then columns, then remove NaNs

# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster

heat_df = df[df['Crime']=='Arson'] # Reducing data size so it runs faster

# heat_df = df[['Latitude', 'Longitude']]

# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])



# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(m)



# Display the map

m
# df_cr = folium.map.FeatureGroup()



# df_cr = plugins.MarkerCluster().add_to(m)



# loop through the crimes and add each to the incidents feature group

# for lat, lng, label in zip(df.Latitude, df.Longitude, df.Crime):

  #  df_cr.add_child(

   #     folium.CircleMarker(

    #        [lat, lng],

     #       radius=5, # define how big you want the circle markers to be

      #      color='yellow',

       #     fill=True,

        #    popup=label,

         #   fill_color='blue',

          #  fill_opacity=0.6

        #)

    #)



# add incidents to map

# m.add_child(df_cr)

# m
df.drop(df[df.CrimeDate > '08-31-2017'].index, inplace=True)
df.groupby(

  pd.Grouper(

    key='CrimeDate',

    freq='M'

  )

).size().plot.line(figsize=(24, 8), linewidth=3.5)
df.groupby(

  pd.Grouper(

    key='CrimeDate',

    freq='M'

  )

).size().plot.bar(figsize=(25, 5))