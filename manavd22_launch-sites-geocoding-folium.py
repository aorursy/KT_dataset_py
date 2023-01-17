# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

from geopandas.tools import geocode

from geopy.geocoders import Nominatim

import folium

from folium.plugins import MarkerCluster
data = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
data.dtypes
data.drop(data.columns[[0,1]],axis=1,inplace=True)
data.dtypes
sns.heatmap(data=pd.isna(data),yticklabels=False,cmap='YlGn')
data.rename(columns={' Rocket':'Rocket'},inplace=True)
data.head()
data.drop(data.columns[[5]],axis=1,inplace=True)
result=pd.DataFrame(columns=['Location','Coordinates'])

result['Location']=data['Location'].unique()
result
result['Location_2']=result['Location'].str.split(',').str[1]
locator = Nominatim(user_agent='myGeocoder')

for i in range(result.shape[0]):

    if locator.geocode(result.loc[i]['Location_2'])!=None:

        result.loc[i,'Coordinates'] = locator.geocode(result.loc[i]['Location_2'])[1]

    else:

        result.loc[i,'Coordinates'] = None
result
data_1=data
data_1=data_1.merge(result,how='left',on='Location')
data_1
data_1[pd.isna(data_1.Coordinates)][['Location']].value_counts()
data_1.loc[data_1.Location_2==' Plesetsk Cosmodrome','Coordinates']=[[62.925644,40.577878]]

data_1.loc[data_1.Location_2==' Kiritimati Launch Area','Coordinates']=[[0,-154]]

data_1.loc[data_1.Location_2==' M?hia Peninsula','Coordinates']=[[-39.15,177.9]]

data_1.loc[data_1.Location_2==' Palmachim Airbase','Coordinates']=[[31.897778,34.690556]]

data_1.loc[data_1.Location_2==' Yasny Cosmodrome','Coordinates']=[[51.05,59.966667]]

data_1.loc[data_1.Location_2==' San Marco Launch Platform','Coordinates']=[[-2.938333,40.2125]]

data_1.loc[data_1.Location_2==' RAAF Woomera Range Complex','Coordinates']=[[-30.9553,136.5322]]

data_1.loc[data_1.Location_2==' Wenchang Satellite Launch Center','Coordinates']=[[19.614354,110.951057]]

data_1.loc[data_1.Location_2==' Alc?›ntara Launch Center','Coordinates']=[[-2.339444,-44.4175]]

data_1.loc[data_1.Location_2==' Shahrud Missile Test Site','Coordinates']=[[36.418056,54.976389]]

data_1.loc[data_1.Location_2==' Barents Sea Launch Area','Coordinates']=[[69.5,34.2]]
sns.heatmap(data=pd.isna(data_1),yticklabels=False,cmap='YlGn')
m = folium.Map(location=[0, 0],zoom_start=2)

mc = MarkerCluster()

for i in range(data_1.shape[0]):

    mc.add_child(folium.Marker(location=list(data_1.loc[i]['Coordinates']),popup = folium.Popup(data_1.loc[i]['Location'])))

m.add_child(mc)