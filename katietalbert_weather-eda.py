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
os.chdir('/kaggle/input/buildingdatagenomeproject2')

os.listdir()
weather = pd.read_csv('weather.csv')

print(weather.shape)

weather.head()
weather.dtypes
weather.info()
weather['timestamp'] = weather['timestamp'].astype('datetime64')
weather.dtypes
weather.site_id.unique()
weather[weather.site_id == 'Panther'].head(3)
weather[weather.site_id == 'Panther'].tail(3)
weather[weather.site_id == 'Robin'].head(3)
weather[weather.site_id == 'Robin'].tail(3)
import missingno as msno

msno.matrix(weather);
weather = weather.drop(['cloudCoverage', 'precipDepth6HR'], axis=1)

weather.head()
clean_weather = weather[weather.site_id == 'Panther'].drop('site_id', axis=1)

clean_weather.head()
clean_weather = clean_weather.rename(columns={"airTemperature": "Panther_airTemperature", "dewTemperature": "Panther_dewTemperature",

                                              "precipDepth1HR": "Panther_precipDepth1HR", "seaLvlPressure": "Panther_seaLvlPressure",

                                              "windDirection": "Panther_windDirection", "windSpeed": "Panther_windSpeed"})

clean_weather.head()
clean_weather.shape
sites_df = weather[weather.site_id != 'Panther']

sites_df.head()
sites = sites_df.site_id.unique()



for site in sites:

    site_weather = sites_df[sites_df.site_id == site]

    site_weather = site_weather.drop('site_id', axis=1)

    site_weather = site_weather.rename(columns={"airTemperature": site+"_airTemperature", "dewTemperature": site+"_dewTemperature",

                                                "precipDepth1HR": site+"_precipDepth1HR", "seaLvlPressure": site+"_seaLvlPressure",

                                                "windDirection": site+"_windDirection", "windSpeed": site+"_windSpeed"})

    clean_weather = pd.merge(left=clean_weather, right=site_weather, how='left', left_on='timestamp', right_on='timestamp')



clean_weather.head()
clean_weather.shape
msno.matrix(clean_weather);
clean_weather.isnull().sum().sort_values(ascending=False).head(10)
drop_cols = ['Robin_precipDepth1HR', 'Mouse_precipDepth1HR', 'Robin_precipDepth1HR', 'Lamb_precipDepth1HR', 'Lamb_seaLvlPressure',

             'Wolf_precipDepth1HR', 'Shrew_precipDepth1HR', 'Crow_precipDepth1HR', 'Moose_precipDepth1HR',

             'Cockatoo_precipDepth1HR']

clean_weather = clean_weather.drop(drop_cols, axis=1)

clean_weather.shape
clean_weather.isnull().sum().sort_values(ascending=False).head(10)
clean_weather = clean_weather.interpolate(method="slinear")

clean_weather.isnull().sum().sort_values(ascending=False).head(10)
msno.matrix(clean_weather);
clean_weather = clean_weather.fillna(method='ffill')

clean_weather.isnull().sum().sort_values(ascending=False).head(10)
clean_weather = clean_weather.fillna(method='bfill')

clean_weather.isnull().sum().sort_values(ascending=False).head(10)
clean_weather.to_csv('/kaggle/working/weather_cleaned.csv')