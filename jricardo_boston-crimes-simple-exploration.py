# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots

import datetime # manipulating date formats



#Maps

import folium

from folium.plugins import HeatMap



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# settings

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline



# Any results you write to the current directory are saved as output.
#Load Data

crimes = pd.read_csv("../input/crime.csv", encoding='Windows-1252')
#Building DF Districts

data = {'DISTRICT':['A1','A15','A7', 'B2','B3', 'C11', 'C6', 'D14', 'D4', 'E13', 'E18', 'E5',''], 'NAMES':['Downtwon','Charlestown','East Boston','Roxbury', 'Mattapan', 'Dorchester', 'South Boston', 'Brighton', 'South End', 

                           'Jamaica Plain', 'Hyde Park', 'West Roxbury', 'Unknown Location']} 

  

# Create DataFrame 

districts = pd.DataFrame(data) 

  

# Print new DF 

districts
#Simple Preview 

crimes.head(5)
districts.head(5)
#Create dataframe with Day Period

hour_day = ['Daybreak']*6 + ['Morning']*6 + ['Evening']*6 + ['Night']*6



hour = [list(range(0,24))]

for cont in hour:

    df = [(cont * 1),hour_day]

    

df2 = pd.DataFrame.from_dict(df)



period = df2.transpose()

period.columns = ["HOUR", "HDAY_NAME"]

period.head()
#Let's add the Column 'District Name'("vlookup" between Crimes and District)

crimes = crimes.merge(districts, on = 'DISTRICT')
period['HOUR']=period['HOUR'].astype(int)
#Let's add the Column 'District Name'("vlookup" between Crimes and Period)

crimes = crimes.merge(period, on = 'HOUR')

crimes.head()
#Some columns not be use in this example. So let's drop two: 



crimes = crimes.drop(["INCIDENT_NUMBER", "OFFENSE_DESCRIPTION"], axis=1)



#other way to delete some columns, could be 

#del crimes['INCIDENT_NUMBER']

#del crimes['OFFENSE_DESCRIPTION']
#Reorder Columns to better visualization.

crimes = crimes[['OCCURRED_ON_DATE','OFFENSE_CODE','OFFENSE_CODE_GROUP','DISTRICT','NAMES','REPORTING_AREA',

                 'SHOOTING', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR','HDAY_NAME', 'UCR_PART','STREET', 'Lat', 'Long', 'Location']]
#Fill NaN values on Shooting

crimes[['SHOOTING']] = crimes[['SHOOTING']].fillna(value='Not Informed')
crimes.head(5)
#View type data

crimes.dtypes
print("\nData size (line x column): {} ".format(crimes.shape)) 
#Applying a crosstab to see general data by Year

view_by_Month = pd.crosstab(crimes["MONTH"], crimes["YEAR"], margins = True)

view_by_Month
crimes['datetime'] = pd.to_datetime(crimes['OCCURRED_ON_DATE'])

crimes = crimes.set_index('datetime')

crimes.drop(['OCCURRED_ON_DATE'], axis=1, inplace=True)

crimes.head()
crimes_2017 = crimes['2017-01-01':'2017-12-31']
#Applying a crosstab to see general data by Month

view_by_Month = pd.crosstab(crimes_2017["MONTH"], crimes_2017["YEAR"], margins = True)

view_by_Month
#Cross by District Name x Month

view_by_District = pd.crosstab(crimes_2017["NAMES"], crimes_2017["MONTH"], margins = True)

view_by_District
#This command drop the column and line sum(do this if necessary)

view_by_District = view_by_District.drop('All',axis=1)

view_by_District = view_by_District.drop('All',axis=0)



view_by_District
#View barplots by District x Month

view_by_District.plot(kind="bar", figsize=(17,6), stacked=True)
#Applying a crosstab to see general data - 12 Months

view_by_Year = pd.crosstab(crimes_2017["YEAR"], crimes_2017["NAMES"], margins = True)

view_by_Year
# NaN Info: replace -1 values in Lat/Long

crimes_2017.Lat.replace(-1, None, inplace=True)

crimes_2017.Long.replace(-1, None, inplace=True)
# Plot districts "Segmentations"

sns.scatterplot(x='Lat', y='Long', hue='NAMES', alpha=0.01,data=crimes_2017)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
grid = sns.FacetGrid(crimes_2017, col='HDAY_NAME', row='YEAR', size=8.7, aspect=1.3,  col_order=["Daybreak", "Morning", "Evening", "Night"])

grid.map(plt.hist, 'MONTH', alpha=.7, bins=12)

grid.add_legend();
#Unified Info by Hour.

plt.subplots(figsize=(27,5))

sns.countplot(x='HOUR', data=crimes_2017)

plt.title('CRIME AND REQUESTS HOURS')
#Applying a crosstab to see general data by Period

view_by_HDAY = pd.crosstab(crimes_2017["HDAY_NAME"], crimes_2017["MONTH"], margins = True)

view_by_HDAY.sort_values(by=['All'], ascending=True)
#Applying a crosstab to see general data by Period

view_by_CrimeGroup = pd.crosstab(crimes_2017["OFFENSE_CODE_GROUP"], crimes_2017["HDAY_NAME"], margins = True)

view_by_CrimeGroup.sort_values(by=['All'], ascending=True)

view_by_CrimeGroup.head(10)
# 10 Main Crimes and Requests/Day Period

view_by_CrimeGroup.nlargest(11, ['All']) 
#Folium crime map

#To other view options, explore documentation tiles.

map_Crime = folium.Map(location=[42.3125,-71.0875], tiles = "OpenStreetMap", zoom_start = 11)



# Add data for heatmp 

heatmap_Info = crimes_2017[crimes_2017.YEAR == 2017]

heatmap_Info = crimes_2017[['Lat','Long']]

heatmap_Info = crimes_2017.dropna(axis=0, subset=['Lat','Long'])

heatmap_Info = [[row['Lat'],row['Long']] for index, row in heatmap_Info.iterrows()]

HeatMap(heatmap_Info, radius=10).add_to(map_Crime)



# Plot Map

map_Crime
#Select a period...

crimes_ts = crimes['2017-01-01':'2017-06-30']
#Delete not relevant columns...

crimes_ts = crimes_ts.drop(["OFFENSE_CODE", "DISTRICT","NAMES","REPORTING_AREA", "SHOOTING","YEAR","MONTH","DAY_OF_WEEK","HOUR","HDAY_NAME",

                            "UCR_PART",	"STREET","Lat","Long","Location"], axis=1)
#"Group by"Dat(D)

ts = crimes_ts.resample('D').count()

ts.head()
ts.plot(figsize=(30,4), grid=True)

plt.title('TIME SERIES: JAN-JUN')