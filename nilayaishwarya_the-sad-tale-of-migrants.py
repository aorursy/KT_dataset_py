import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import datetime

import re

import os

from mpl_toolkits.basemap import Basemap



print('*'*50)

#print('Python Version    : ', sys.version)

print('Pandas Version    : ', pd.__version__)

print('Numpy Version     : ', np.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)

print('*'*50)



sns.set_style('white')



pd.options.display.max_rows = 100

pd.options.display.max_columns = 100
def missingData(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    #print(total)

    #print(percent)

    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print(md)

    md = md[md["Percent"] > 0]

    print(md)

    plt.figure(figsize = (8, 4))

    plt.xticks(rotation='90')

    sns.barplot(md.index, md["Percent"],color="g",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return md



def valueCounts(dataset, features):

    """Display the features value counts """

    for feature in features:

        vc = dataset[feature].value_counts()

        print(vc)

        print('-'*30)
inp_data = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')

inp_data.head()
missingData(inp_data)
f = ['Cause of Death']

valueCounts(inp_data, f)

data=inp_data
data['Lat'], data['Lon'] = data['Location Coordinates'].str.split(', ').str

data.Lat=data.Lat.astype(float)

data.Lon=data.Lon.astype(float)

#print(data['Lat'])

#print(data['Lon'])
def convert_date(s):

    new_s = datetime.datetime.strptime(s, '%B %d, %Y')

    return new_s
data['Date'] = data['Reported Date'].apply(convert_date)
#print(data['Number Dead'])

data['Number Dead'].fillna(0, inplace=True)

#print(data['Number Dead'])
data['Minimum Estimated Number of Missing'].fillna(0, inplace=True)
data['Total Dead and Missing'] = data['Total Dead and Missing'].astype(float)
toDrop = [

    'Reported Date',

    'Web ID',

    'Number of Survivors',

    'Location Coordinates',

    'URL',

    'UNSD Geographical Grouping'

]

data.drop(toDrop, axis=1, inplace=True)

data.shape

data.head()
def deathCauseReplacement(data):

    #HEALTH CONDITION

    data.loc[data['Cause of Death'].str.contains('Sickness|sickness'), 'Cause of Death'] = 'Health Condition'

    data.loc[data['Cause of Death'].str.contains('diabetic|heart attack|meningitis|virus|cancer|bleeding|insuline|inhalation'), 'Cause of Death'] = 'Health Condition'

    data.loc[data['Cause of Death'].str.contains('Organ|Coronary|Envenomation|Post-partum|Respiratory|Hypoglycemia'), 'Cause of Death'] = 'Health Condition'

    #HARSH CONDITIONS

    data.loc[data['Cause of Death'].str.contains('harsh weather|Harsh weather'), 'Cause of Death'] = 'Harsh conditions'

    data.loc[data['Cause of Death'].str.contains('Harsh conditions|harsh conditions'), 'Cause of Death'] = 'Harsh conditions'

    data.loc[data['Cause of Death'].str.contains('Exhaustion|Heat stroke'), 'Cause of Death'] = 'Harsh conditions'

    #UNKNOWN

    data.loc[data['Cause of Death'].str.contains('Unknown|unknown'), 'Cause of Death'] = 'Unknown'

    #STARVATION

    data.loc[data['Cause of Death'].str.contains('Starvation|starvation'), 'Cause of Death'] = 'Starvation'

    #DEHYDRATION

    data.loc[data['Cause of Death'].str.contains('dehydration|Dehydration'), 'Cause of Death'] = 'Dehydration'

    #DROWNING

    data.loc[data['Cause of Death'].str.contains('Drowning|drowning|Pulmonary|respiratory|lung|bronchial|pneumonia|Pneumonia'), 'Cause of Death'] = 'Drowning'

    #HYPERTHERMIA

    data.loc[data['Cause of Death'].str.contains('hyperthermia|Hyperthermia'), 'Cause of Death'] = 'Hyperthermia'

    #HYPOTHERMIA

    data.loc[data['Cause of Death'].str.contains('hypothermia|Hypothermia'), 'Cause of Death'] = 'Hypothermia'

    #ASPHYXIATION

    data.loc[data['Cause of Death'].str.contains('asphyxiation|suffocation'), 'Cause of Death'] = 'Asphyxiation'

    #VEHICLE ACCIDENT

    data.loc[data['Cause of Death'].str.contains('train|bus|vehicle|truck|boat|car|road|van|plane'), 'Cause of Death'] = 'Vehicle Accident'

    data.loc[data['Cause of Death'].str.contains('Train|Bus|Vehicle|Truck|Boat|Car|Road|Van|Plane'), 'Cause of Death'] = 'Vehicle Accident'

    #MURDER

    data.loc[data['Cause of Death'].str.contains('murder|stab|shot|violent|blunt force|violence|beat-up|fight|murdered|death'), 'Cause of Death'] = 'Murder'

    data.loc[data['Cause of Death'].str.contains('Murder|Stab|Shot|Violent|Blunt force|Violence|Beat-up|Fight|Murdered|Death'), 'Cause of Death'] = 'Murder'

    data.loc[data['Cause of Death'].str.contains('Hanging|Apache|mortar|landmine|Rape|Gassed'), 'Cause of Death'] = 'Murder'

    #CRUSHED

    data.loc[data['Cause of Death'].str.contains('crushed to death|crush|Crush|Rockslide'), 'Cause of Death'] = 'Crushed'

    #BURNED

    data.loc[data['Cause of Death'].str.contains('burn|burns|burned|fire'), 'Cause of Death'] = 'Burned'

    data.loc[data['Cause of Death'].str.contains('Burn|Burns|Burned|Fire'), 'Cause of Death'] = 'Burned'

    #ELECTROCUTION

    data.loc[data['Cause of Death'].str.contains('electrocution|Electrocution'), 'Cause of Death'] = 'Electrocution' #folgorazione

    #FALLEN

    data.loc[data['Cause of Death'].str.contains('Fall|fall'), 'Cause of Death'] = 'Fallen' 

    #KILLED BY ANIMALS

    data.loc[data['Cause of Death'].str.contains('crocodile|hippopotamus|hippoptamus'), 'Cause of Death'] = 'Killed by animals'

    #EXPOSURE

    data.loc[data['Cause of Death'].str.contains('exposure|Exposure'), 'Cause of Death'] = 'Exposure'
deathCauseReplacement(data)
valueCounts(data,['Cause of Death'])
fig = plt.figure(figsize=(20, 14)) 

data['Cause of Death'].value_counts().plot(kind='bar', 

                                   color='b',

                                   align='center')

plt.title('Cause of Death', fontsize=20)

plt.show()
fig = plt.figure(figsize=(20, 14)) 

data['Region of Incident'].value_counts().plot(kind='bar', 

                                   color='y',

                                   align='center')

plt.title('Region Of Incident', fontsize=20)

plt.show()
lat = data['Lat'][:]

lon = data['Lon'][:]

lat = lat.dropna()

lon = lon.dropna()

lat = np.array(lat)

lon = np.array(lon)



fig=plt.figure()

ax=fig.add_axes([1.0,1.0,2.8,2.8])

mapp = Basemap(llcrnrlon=-180.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=80.,

            rsphere=(6378137.00,6356752.3142),

            resolution='l',projection='merc',

            lat_0=40.,lon_0=-20.,lat_ts=20.)

mapp.drawcoastlines()

mapp.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])

mapp.drawmeridians(np.arange(mapp.lonmin,mapp.lonmax+30,60),labels=[0,0,0,1])

x, y = mapp(lon,lat)

mapp.scatter(x,y,3,marker='o',color='r')

ax.set_title('Refugee deaths across the world', fontsize=20)

plt.show()
data.pivot_table('Total Dead and Missing', index='Migration Route', columns='Reported Year', aggfunc='sum')
data.pivot_table(['Number of Males','Number of Females','Number of Children'], 

                 index='Reported Year',

                 aggfunc={'Number of Males': np.sum,'Number of Females': np.sum,'Number of Children': np.sum}).plot(figsize=(20, 10), kind='bar')

plt.ylabel('Count')

plt.title('Total Dead and Missing by Sex and Age', fontsize=20)

plt.show()
data.pivot_table('Number of Children', 

                 index='Cause of Death',

                 aggfunc=np.sum).plot(figsize=(20, 10), kind='bar')

plt.ylabel('Count')

plt.title('Total Children Deaths and thier causes', fontsize=20)

plt.show()
#def deathChildrenDrowning(data):

    #DROWNING

 #   data.loc[data['Cause of Death'].str.contains('Drowning'), 'Cause of Death'] = 'Health Condition'

    
cdr=data[data['Cause of Death'].str.contains('Drowning')]

#cdr

cdr.pivot_table('Number of Children', 

                 index='Region of Incident',

                 aggfunc=np.sum).plot(figsize=(20, 10), kind='bar')

plt.ylabel('Count')

plt.title('Total Children Deaths due to drowning in different regions', fontsize=20)

plt.show()
datachild=data[data['Number of Children']>0]

lat = datachild['Lat'][:]

lon = datachild['Lon'][:]

lat = lat.dropna()

lon = lon.dropna()

lat = np.array(lat)

lon = np.array(lon)



fig=plt.figure()

ax=fig.add_axes([1.0,1.0,2.8,2.8])

mapp = Basemap(llcrnrlon=-180.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=80.,

            rsphere=(6378137.00,6356752.3142),

            resolution='l',projection='merc',

            lat_0=40.,lon_0=-20.,lat_ts=20.)

mapp.drawcoastlines()

mapp.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])

mapp.drawmeridians(np.arange(mapp.lonmin,mapp.lonmax+30,60),labels=[0,0,0,1])

x, y = mapp(lon,lat)

mapp.scatter(x,y,3,marker='o',color='b')

ax.set_title('Children deaths across the world', fontsize=20)

plt.show()