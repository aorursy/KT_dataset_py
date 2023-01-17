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
data = pd.read_csv('/kaggle/input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')

data.head()
data.info()
data.describe()
data.shape
# data.iloc[data.isnull().sum()>0]

data[data.isnull().sum(axis = 1)>0]
data.PdDistrict.value_counts()
data['PdDistrict'].fillna(data['PdDistrict'].mode()[0], inplace = True)



data[data.isnull().sum(axis = 1)>0]
# #Missing Values

# from sklearn.experimental import enable_iterative_imputer

# from sklearn.impute import IterativeImputer

# help(IterativeImputer)
# it = IterativeImputer()

# new_data = it.fit_transform(data)

# new_data
import datetime as dt
data.dtypes
data.Date = pd.to_datetime(data.Date)

data.Date
data.Time = pd.to_datetime(data.Time).dt.time

data.Time
data.info()
data.head()
# col_objects = ['Category','Descript','DayOfWeek','Time','PdDistrict','Resolution','Address','Location']



# for col in col_objects:

#     data[col] = data[col].astype(str)

# data.info()
import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = (20,9)

plt.style.use('dark_background')



sns.countplot(data['Category'],palette = 'gnuplot')



plt.title('Major Crimes in Sanfrancisco', fontweight = 30, fontsize =20)

plt.xticks(rotation = 90)

plt.show()
data.Descript.value_counts(normalize=True)
data.DayOfWeek.value_counts(normalize=True)
sns.catplot(data = data, x = 'DayOfWeek', kind = 'count',palette = 'gnuplot', aspect = 1.8,

            order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])



plt.title('Day of the week', fontweight = 30, fontsize =20)

# plt.xticks(rotation = 90)

plt.show()
data.Time.value_counts().plot.line()

plt.show()
sns.catplot(data = data, x = 'PdDistrict', kind = 'count',palette = 'gnuplot', aspect = 2.8)



plt.title('District', fontweight = 30, fontsize =20)

# plt.xticks(rotation = 90)

plt.show()
data.Resolution.value_counts()
import folium
# San Francisco latitude and longitude values

latitude = 37.77

longitude = -122.42



# create map and display it

sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)



# display the map of San Francisco

sanfran_map
# get the first 100 crimes in the df_incidents dataframe

limit = 100

df_incidents = data.iloc[0:limit, :]



# instantiate a feature group for the incidents in the dataframe

incidents = folium.map.FeatureGroup()



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, in zip(df_incidents.Y, df_incidents.X):

    incidents.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



# add incidents to map

sanfran_map.add_child(incidents)

sanfran_map
# instantiate a feature group for the incidents in the dataframe

incidents = folium.map.FeatureGroup()



df_incidents = data



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, in zip(df_incidents.Y, df_incidents.X):

    incidents.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



# add pop-up text to each marker on the map

latitudes = list(df_incidents.Y)

longitudes = list(df_incidents.X)

labels = list(df_incidents.Category)



for lat, lng, label in zip(latitudes, longitudes, labels):

    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)    

    

# add incidents to map

sanfran_map.add_child(incidents)

sanfran_map
data.PdDistrict.value_counts(normalize=True)
data.PdDistrict.value_counts(normalize=True).plot.pie()
data.groupby('PdDistrict')['PdId'].count().plot(kind = 'bar')