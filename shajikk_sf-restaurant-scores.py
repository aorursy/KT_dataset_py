# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv(r'../input/restaurant-scores-lives-standard.csv')
#df.head()
df_unique = df.drop_duplicates(subset='business_name', keep="first")
#df_unique.head()
df_unique = df_unique[['business_name', 'business_latitude', 'business_longitude', 'risk_category']]


df_unique = df_unique.dropna(subset=['business_latitude', 'business_longitude'])
df_unique = df_unique.replace(np.nan, 'unknown', regex=True)
#df_unique.head()
#risks = df_unique.risk_category.unique()
risks = {
    'Low Risk'      : 'green', 
    'Moderate Risk' : 'blue', 
    'High Risk'     : 'red', 
    'unknown'       : 'grey',
    }

df.shape

import folium

df_sf = df_unique.head(500) # Display only 500

city_location = (37.76, -122.45)

sf_map = folium.Map(location=city_location, zoom_start=12) 

def plotAddress(df):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    #print("%s" %(risks[df.risk_category]))
    marker_color = risks[df.risk_category]
    folium.CircleMarker(
                        location=[df.business_latitude, df.business_longitude],
                        radius=3,
                        weight=3,
                        popup=df.business_name,
                        fill=True,
                        fill_color=marker_color,
                        fill_opacity=1,
                        color=marker_color, 
                       ).add_to(sf_map)


df_sf.apply(plotAddress, axis = 1)


display(sf_map)
