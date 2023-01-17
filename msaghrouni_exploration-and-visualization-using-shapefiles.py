import os
print(os.listdir("../"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# import dataset
accidents = pd.read_csv("../input/dft-accident-data/Accidents0515.csv")

# take a look at first entries
accidents.head()
# number of rows by number of columns
accidents.shape
# check for 'NaN' values
accidents.isnull().sum()
# fill the missing values for time with 'N/A'
accidents['Time'].fillna('N/A', inplace = True)
# fill the missing values for Longitude with 999
accidents['Longitude'].fillna(999, inplace = True)
# fill the missing values for Latitude with 999
accidents['Latitude'].fillna(999, inplace = True)
# Create a function to categorize accident severity
def category(accident_severity):
    if accident_severity == 1:
        return 'Fatal'
    elif accident_severity == 2:
        return 'Serious'
    elif accident_severity == 3:
        return 'Slight'
    else:
        return 'Unknown'
# Apply the function and add it as 'Accident_Category' column
accidents['Accident_Category'] = accidents['Accident_Severity'].apply(category)
# Change the index to the column 'Date'
accidents.index = pd.DatetimeIndex(accidents['Date'])
plt.figure(figsize=(15,6))
plt.title('Distribution of accidents per day', fontsize=16)
plt.tick_params(labelsize=14)
sns.distplot(accidents.resample('D').size(), bins=60);
# Create an Upper Control Limit (UCL) and a Lower Control Limit (LCL)
accidents_daily = pd.DataFrame(accidents.resample('D').size())
accidents_daily['MEAN'] = accidents.resample('D').size().mean()
accidents_daily['STD'] = accidents.resample('D').size().std()
UCL = accidents_daily['MEAN'] + 3 * accidents_daily['STD']
LCL = accidents_daily['MEAN'] - 3 * accidents_daily['STD']

# Plot total accidents per day, UCL, LCL and moving-average
plt.figure(figsize=(15,6))
accidents.resample('D').size().plot(label='Accidents per day')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
accidents_daily['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Total accidents per day', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Number of accidents')
plt.tick_params(labelsize=14)
plt.legend(prop={'size':16})
# convert the string 'Date' to date
accidents['convert_to_date'] = pd.to_datetime(accidents['Date'])
# add column 'Day', 'Month', 'Year' to the dataframe
accidents['Day'] = accidents['convert_to_date'].dt.day
accidents['Month'] = accidents['convert_to_date'].dt.month
accidents['Year'] = accidents['convert_to_date'].dt.year
# Create a pivot table by crossing the day number by the month and calculate the average number of accidents for each crossing
accidents_pivot_table = accidents.pivot_table(values='Date', index='Day', columns='Month', aggfunc=len)
accidents_pivot_table_date_count = accidents.pivot_table(values='Date', index='Day', columns='Month', aggfunc=lambda x: len(x.unique()))
accidents_average = accidents_pivot_table/accidents_pivot_table_date_count
accidents_average.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Using seaborn heatmap
plt.figure(figsize=(7,9))
plt.title('Average number of accidents per day and month', fontsize=14)
sns.heatmap(accidents_average.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
# exclude the rows with 'N/A' values in 'Time' column
accidents_time_not_null = accidents[accidents['Time'] != 'N/A']

# convert the string 'Time' to time and extract the hour
accidents_time_not_null['Hour'] = pd.to_datetime(accidents_time_not_null['Time'], format='%H:%M').dt.hour
# Create a pivot table by crossing the hour by the day of the week and calculate the average number of accidents for each crossing
accidents_pivot_table = accidents_time_not_null.pivot_table(values='Date', index='Hour', columns='Day_of_Week', aggfunc=len)
accidents_pivot_table_date_count = accidents_time_not_null.pivot_table(values='Date', index='Hour', columns='Day_of_Week', aggfunc=lambda x: len(x.unique()))
accidents_average = accidents_pivot_table/accidents_pivot_table_date_count
accidents_average.columns = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

# Using seaborn heatmap
plt.figure(figsize=(6,6))
plt.title('Average number of accidents per hour and day of the week', fontsize=14)
plt.tick_params(labelsize=12)
sns.heatmap(accidents_average.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
# Using resample 'M' and rolling window 12
plt.figure(figsize=(15,6))
accidents.resample('M').size().plot(label='Total per month')
accidents.resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')

plt.title('Accidents per month', fontsize=16)
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)
# Using pivot_table to groub by date and category, resample 'M' and rolling window 12
accidents.pivot_table(values='Accident_Severity', index='convert_to_date', columns='Accident_Category', aggfunc=len).resample('M').sum().rolling(window=12).mean().plot(figsize=(15,6), linewidth=4)
plt.title('Moving average of accidents per month by accident category', fontsize=16)
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16)
# Create a pivot table
accidents_pivot_table = accidents.pivot_table(values='Accident_Severity', index='Accident_Category', columns='Month', aggfunc=len)

# Scale each category by diving by the max value of each one
accidents_scaled = pd.DataFrame(accidents_pivot_table.iloc[0] / accidents_pivot_table.iloc[0].max())
for i in [2,1]:
    accidents_scaled[accidents_pivot_table.index[i]] =  pd.DataFrame(accidents_pivot_table.iloc[i] / accidents_pivot_table.iloc[i].max())

# Using seaborn heatmap
plt.figure(figsize=(7,9))
plt.title('Heatmap of accident_severity by month', fontsize=14)
sns.heatmap(accidents_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False)
# Create a pivot table
accidents_pivot_table = accidents.pivot_table(values='Accident_Severity', index='Accident_Category', columns='Day_of_Week', aggfunc=len)

# Scale each category by diving by the max value of each one
accidents_scaled = pd.DataFrame(accidents_pivot_table.iloc[0] / accidents_pivot_table.iloc[0].max())
for i in [2,1]:
    accidents_scaled[accidents_pivot_table.index[i]] =  pd.DataFrame(accidents_pivot_table.iloc[i] / accidents_pivot_table.iloc[i].max())

accidents_scaled.index = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

# Using seaborn heatmap
plt.figure(figsize=(7,9))
plt.title('Heatmap of accident_severity by day of week', fontsize=14)
sns.heatmap(accidents_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False)
# Create a pivot table
accidents_pivot_table = accidents_time_not_null.pivot_table(values='Accident_Severity', index='Accident_Category', columns='Hour', aggfunc=len)
  
# Scale each category by diving by the max value of each one
accidents_scaled = pd.DataFrame(accidents_pivot_table.iloc[0] / accidents_pivot_table.iloc[0].max())
for i in [2,1]:
    accidents_scaled[accidents_pivot_table.index[i]] =  pd.DataFrame(accidents_pivot_table.iloc[i] / accidents_pivot_table.iloc[i].max())


# Using seaborn heatmap
plt.figure(figsize=(7,9))
plt.title('Heatmap of accident_severity by hour', fontsize=14)
sns.heatmap(accidents_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False)
# Prepare the data
accidents2 = accidents[accidents['Longitude'] != 999]
from sklearn.utils import shuffle

# shuffle the data
accidents2 = shuffle(accidents2)
import shapefile

# read the shapefile
data_in_shapefile = shapefile.Reader('../input/distribution/Areas')
print(data_in_shapefile.numRecords)
from shapely.geometry import shape

# Obtain the attributes and the geometry for each record.
attributes, geometry = [], []
field_names = [field[0] for field in data_in_shapefile.fields[1:]]  
for row in data_in_shapefile.shapeRecords():  
    geometry.append(shape(row.shape.__geo_interface__))  
    attributes.append(dict(zip(field_names, row.record))) 
print (attributes)
import geopandas as gpd

# populate a geopandas dataframe
gdf = gpd.GeoDataFrame(data = attributes, geometry = geometry)
gdf.head()
from shapely.geometry import Point

# create a function to map the accidents coordinates
def map_accidents_to_areas(longitude, latitude):       
        point = (float(longitude),float(latitude))
        for i in range(len(gdf.geometry)):            
            if Point(point).within(gdf.geometry.loc[i]):
                return gdf.name[i]    
import time

start = time.clock()
acc = accidents2.iloc[1:300000,:]
acc['Area'] = acc.apply(lambda x: map_accidents_to_areas(x['Longitude'], x['Latitude']), axis=1)
end = time.clock()
print ("%.2gs" % (end-start))
# create a new dataframe by grouping by 'Area' and counting the number of accidents
acc_by_area = pd.DataFrame({'Total_accidents' : acc.groupby( ["Area"] ).size()}).reset_index()
# create a lookup function
def find_total_accidents(area_name):
    for i in range(len(acc_by_area.Area)):
        if area_name == acc_by_area.Area.loc[i]:
                return acc_by_area.Total_accidents[i]    
# add a new column to the geopandas dataframe
gdf['Total_accidents'] = gdf.name.apply(find_total_accidents)
import matplotlib.cm
from matplotlib.colors import Normalize

vmin, vmax = gdf['Total_accidents'].min(), gdf['Total_accidents'].max()

# create a Choropleth map (map where the color of each shape is based on the value of an associated variable)
ax = gdf.plot(column='Total_accidents', cmap='OrRd', edgecolor='black', figsize=(12,12), linewidth=1)
fig = ax.get_figure()
# create a ScalarMappable object and use the set_array() function to add the accidents counts to it
sm = matplotlib.cm.ScalarMappable(cmap='OrRd', norm=Normalize(vmin=vmin, vmax=vmax))
sm.set_array(gdf['Total_accidents'])
# Then pass it to the colorbar() function and set the shrink argument to 0.4 in order to make the colorbar smaller than the map
fig.colorbar(sm, shrink=0.4)