# Importing libs



# for numbers



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for visualisation



import folium

from folium import plugins

from folium.plugins import HeatMap

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
# Taking dataframe from CSV file



df = pd.read_csv('/kaggle/input/arrests-data-by-new-york-police-department/NYPD_Arrests_Data__Historic_.csv', dtype={'PD_CD':str, 'KY_CD':str, 'JURISDICTION_CODE':str})

df.info()
df.shape
# Return column list



df.columns
df = df.rename(columns={'ARREST_DATE':'Date', 'ARREST_KEY':'ID', 'PD_CD':'PDcode', 'PD_DESC':'PDdesc', 'KY_CD':'KYdesc', 'OFNS_DESC':'Desc', 'LAW_CODE':'CoLaw', 'LAW_CAT_CD':'Oflevel', 'ARREST_BORO':'Boro', 'ARREST_PRECINCT':'Precinct', 'JURISDICTION_CODE':'Jurisdiction', 'AGE_GROUP':'Age', 'PERP_SEX':'Sex', 'PERP_RACE':'Race'}, index={'ARREST_DATE': 'Date'})

df = df.drop(columns={'X_COORD_CD', 'Y_COORD_CD'})

df
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

df['Date'].value_counts()
df['Boro'] = df['Boro'].replace({'Q': 'Queens', 'M': 'Manhattan', 'S': 'Staten Island', 'B': 'Bronx', 'K': 'Brooklyn'})

df['Sex'] = df['Sex'].replace({'M': 'Male', 'F': 'Female'})

df['Oflevel'] = df['Oflevel'].replace({'F': 'Felony', 'M': 'Misdemeanor', 'V': 'Violation'})

df['Jurisdiction'] = df['Jurisdiction'].replace({'0': 'Patrol', '1': 'Transit', '2': 'Housing'})
df['PDdesc'] = df['PDdesc'].str.capitalize()

df['Race'] = df['Race'].str.title()

df['Desc'] = df['Desc'].str.capitalize()
df.head(3)
df['Desc'].value_counts().head(15).plot.pie(radius=2.7, autopct='%1.1f%%', textprops=dict(color="black"))
df['Boro'].value_counts().head(15).plot.pie(radius=2.1, autopct='%1.1f%%', textprops=dict(color="black"))
m = folium.Map(location=[40.7221, -73.9198], zoom_start=11)
hm_race = df[df['Race']=='White']

hm_race.tail(5)
df = df[pd.notnull(df['Latitude'])]
df = df[pd.notnull(df['Longitude'])]
# Ensure you're handing it floats

df['Latitude'] = df['Latitude'].astype(float)

df['Longitude'] = df['Longitude'].astype(float)



# Filter the DF for rows, then columns, then remove NaNs

# heat_df = df[df['CrimeDate']=='2015-04-27'] # Reducing data size so it runs faster

# heat_df = df[df['Crime']=='Homicide'] # Reducing data size so it runs faster

hm_pol = df[df['Jurisdiction']=='Transit']

# heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])



# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in hm_pol.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(m)



# Display the map

m