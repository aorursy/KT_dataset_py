#Importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

%matplotlib inline

#Loading Dataset

url = '../input/data-analysis-of-911-emergency-calls/911.csv'

data = pd.read_csv(url, header='infer')
data.shape
#checking for null / missing values

data.isna().sum()
#dropping index with missing or null values

data = data.dropna()

data.reset_index(inplace=True, col_level=1, drop=True)
data.head()
data = data.drop(columns=['desc','zip','e'], axis=1)
# Function to extract emergency type

def extract_emergency_type(txt):

    #res = re.findall(r"^\w+",txt)

    result = ''.join(re.findall(r"^\w+",txt))

    return result



#Applying the function to column = title

data['type'] = data['title'].apply(extract_emergency_type)
data.head()
# drop column = title

data = data.drop(columns='title',axis=1)

# rename the column = twp

data = data.rename(columns={'twp':'area'})



# re-arrange the columns

data = data[['timeStamp','type','area','addr','lat','lng']]



#changing to lower case

data['type'] = data['type'].str.lower()

data['area'] = data['area'].str.lower()

data['addr'] = data['addr'].str.lower()
data.head()
# -- Processing the TimeStamp column



data.timeStamp = pd.to_datetime(data.timeStamp, format='%Y-%m-%d %H:%M:%S')

data['year'] = data.timeStamp.apply(lambda x: x.year)

data['month'] = data.timeStamp.apply(lambda x: x.month)

data['day'] = data.timeStamp.apply(lambda x: x.day)

data['hour'] = data.timeStamp.apply(lambda x: x.hour)

data.head()
#Taking backup

data_backup = data.copy()
# Emergencies per Month in 2016

emrgncy_monthly = pd.DataFrame(data[data['year']==2016].groupby('month').size())

emrgncy_monthly['MEAN'] = data[data['year']==2016].groupby('month').size().mean()



plt.figure(figsize=(18,6))

data[data['year']==2016].groupby('month').size().plot(label='Emergencies per month')

emrgncy_monthly['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Total Monthly Emergencies in 2016', fontsize=14)

plt.xlabel('Month')

plt.xticks(np.arange(1,9))

plt.ylabel('Number of emergencies')

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})
# Emergency Types per Month in 2016

emrgncy_type_yrly = data[data['year']==2016].pivot_table(values='area', index='month', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Emergency Types per Month in 2016', fontsize=16)

plt.xlabel('Month')

plt.xticks(np.arange(1,9))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
#Daily Emergencies in July 2016

emrgncy_daily = pd.DataFrame(data[(data['year']==2016) & (data['month']==7)].groupby('day').size())

emrgncy_daily['MEAN'] = data[(data['year']==2016) & (data['month']==7)].groupby('day').size().mean()



plt.figure(figsize=(18,6))

data[(data['year']==2016) & (data['month']==7)].groupby('day').size().plot(label='Emergencies per day in July')

emrgncy_daily['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Daily Emergencies in July 2016', fontsize=14)

plt.xlabel('Day')

plt.xticks(np.arange(1,31))

plt.ylabel('Number of emergencies')

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})
# Emergency Types in month of July 2016

emrgncy_type_mnthly = data[(data['year']==2016) & (data['month']==7)].pivot_table(values='area', index='day', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Emergency Types in month of July 2016', fontsize=16)

plt.xlabel('Day')

plt.xticks(np.arange(1,31))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
#Hourly Emergencies on 25 July 2016

emrgncy_hourly = pd.DataFrame(data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].groupby('hour').size())

emrgncy_hourly['MEAN'] = data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].groupby('hour').size().mean()



plt.figure(figsize=(18,6))

data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].groupby('hour').size().plot(label='Emergencies on 25 July 2016')

emrgncy_hourly['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')

plt.title('Hourly Emergencies on 25 July 2016', fontsize=14)

plt.xlabel('Hours')

plt.xticks(np.arange(1,24))

plt.ylabel('Number of emergencies')

plt.tick_params(labelsize=10)

plt.legend(prop={'size':10})
# Emergency Types on 25th July 2016

emrgncy_type_hourly = data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].pivot_table(values='area', index='hour', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Emergency Types on 25th July 2016', fontsize=16)

plt.xlabel('Hours')

plt.xticks(np.arange(1,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)





# Emergency Types on 13th July 2016

emrgncy_type_hourly = data[(data['year']==2016) & (data['month']==7) & (data['day']==13)].pivot_table(values='area', index='hour', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)

plt.title('Emergency Types on 13th July 2016', fontsize=16)

plt.xlabel('Hours')

plt.xticks(np.arange(1,24))

plt.legend(prop={'size':10})

plt.tick_params(labelsize=10)
# Emergencies per Type During New Year Time [Dec 2015- Jan 2016]



#creating a specific dataframe

ny_df = data[(data['year'].isin(['2015','2016'])) & (data['month'].isin(['12','1']))]



ny_emergencies = ny_df.pivot_table(values='area', index='type', columns=['month','year'], aggfunc=len)

ny_emergencies.columns = ['Jan-2016','Dec-2015']





# Using seaborn heatmap

plt.figure(figsize=(6,6))

plt.title('Emergencies per Type During New Year Time [Dec 2015- Jan 2016]', fontsize=14)

plt.tick_params(labelsize=10)

sns.heatmap(ny_emergencies, cmap='icefire', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")
#Creating a pivot table for Year = 2016

percChng_2016 = data[data['year']==2016].pivot_table(values='area', index='month', columns='type', aggfunc=len)
#Calculating the percentage change

percChng_2016 = percChng_2016.pct_change()
#dropping the index with NULL values

percChng_2016 = percChng_2016.dropna()
# Plotting the Heatmap - Percentage Change in Emergency Calls per Type in 2016

plt.figure(figsize=(6,6))

plt.title('Percentage Change in Emergency Calls per Type in 2016', fontsize=14)

plt.tick_params(labelsize=10)

ax = sns.heatmap(percChng_2016, cmap='Blues', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".2%")

# Geovisualization library

import folium

from folium.plugins import CirclePattern, HeatMap, HeatMapWithTime, FastMarkerCluster
# Function to create a base map



def generateBaseMap(default_location=[40.0655815,-75.2430282], default_zoom_start=10):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start,width='50%', height='50%', tiles="cartodbpositron")

    return base_map



#Calling the function

base_map = generateBaseMap()
# Spatial Visualisation of Emergencies on 25 Jul 2016

spt_df_25Jul2016 = data[(data['year']==2016) & (data['month']==7) & (data['day']==25)]



for i in range(0,len(spt_df_25Jul2016)):

    

    if spt_df_25Jul2016.iloc[i]['type'] == 'traffic':

        folium.CircleMarker(location=[spt_df_25Jul2016.iloc[i]['lat'], spt_df_25Jul2016.iloc[i]['lng']],

                      popup=spt_df_25Jul2016.iloc[i]['type'],radius = 2, color='#2ca25f',fill=True,    #Green

                      fill_color='#2ca25f').add_to(base_map)

    elif spt_df_25Jul2016.iloc[i]['type'] == 'ems':

        folium.CircleMarker(location=[spt_df_25Jul2016.iloc[i]['lat'], spt_df_25Jul2016.iloc[i]['lng']],

                      popup=spt_df_25Jul2016.iloc[i]['type'],radius = 2, color='#2b8cbe',fill=True,   #Blue 

                      fill_color='#2b8cbe').add_to(base_map)

    else:

         folium.CircleMarker(location=[spt_df_25Jul2016.iloc[i]['lat'], spt_df_25Jul2016.iloc[i]['lng']],

                      popup=spt_df_25Jul2016.iloc[i]['type'],radius = 2, color='#f03b20',fill=True,   #Red

                      fill_color='#f03b20').add_to(base_map)       

    

#Calling the base_map function

base_map
# Addiing Cluster Layer 

FastMarkerCluster(data=list(zip(spt_df_25Jul2016['lat'].values, spt_df_25Jul2016['lng'].values))).add_to(base_map)

folium.LayerControl().add_to(base_map)



#Calling the function

base_map