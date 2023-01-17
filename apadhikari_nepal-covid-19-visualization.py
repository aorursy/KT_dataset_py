# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nepdf=pd.read_csv('/kaggle/input/nepal-covid19-infection-data/nepal_covid.csv')
nepalgeo=r'/kaggle/input/nepal-json-file/nepal.json'

nepal_map=folium.Map(location=[28.4,84.1],zoom_start=6.5,tiles='cartodbpositron')
myscale = (nepdf['Confirmed'].quantile((0,0.1,0.6,0.8,0.9,0.95,0.98,1))).tolist()

nepal_map.choropleth(

    geo_data=nepalgeo,

    data=nepdf,

    columns=['District', 'Confirmed'],

    key_on='feature.properties.DISTRICT',

    fill_color='OrRd',

    threshold_scale=myscale,

    fill_opacity=1, 

    line_opacity=0.4,

    legend_name='Nepal Covid-19 Infection',

    reset=True

)
for i in range (len(nepdf)):

    folium.CircleMarker(

            [nepdf.Y[i],nepdf.X[i]],

            radius=5,

            color='',

            fill=True,

            popup=nepdf['District'][i]+"\n"+"Confirmed:"+str(nepdf['Confirmed'][i])+"\n"+"Death:"+str(nepdf['Death'][i])+"\n"+"Recovered:"+str(nepdf['Recovered'][i]),

            fill_color='red',

            fill_opacity=0.6

        ).add_to(nepal_map)

nepal_map