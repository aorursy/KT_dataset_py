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


# This notebook creates an HTML output PandemicSpread.html. It tracks the global spread of the pandemic. 

# It highlights the time window when the pandemic significantly flared up and spread like wild fire.

# It also visually highlights that the virus is season agnostic affecting both hemispheres simultaneously.



import numpy as np

import pandas as pd

import folium  #for graphing

from folium.plugins import TimestampedGeoJson

from IPython.display import display, HTML



#url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

file = '/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv'

# Read CSV file from file

cases_df = pd.read_csv(file , index_col = 0)

print(cases_df)



# Determine First infection date for each entry and store it as a List

num_cols = len(cases_df.columns) + 1

firstInfectionDates = []

first_infection = ' '

i = 0

for idx , value in cases_df.iterrows():

    

    first_infection = ' '

    for i in range(4 , num_cols): #Date recordings from column 5 onwards - one column per date

       if value[i] > 0:

            first_infection = cases_df.columns[i]

            firstInfectionDates.append(first_infection)

            break # Capture first date recorded and leave





#Convert the List of Dates into a Series

Date = pd.Series(firstInfectionDates , name = 'Date')

# Convert Date to date/time format

Date = pd.to_datetime(Date)







# Slice Data Frame and Combine with the Date series into a new data frame

cases1_df = cases_df[['Country/Region' , 'Lat', 'Long']]

Globalcases_df = pd.concat([cases1_df.reset_index(), Date], axis = 1 )





# Now sort by date to organise chronologically

Globalcases_df.sort_values(['Date'], inplace = True)



#Extract Latitude,Longitude for dropping markers

Mapdata = [[value.Date , value.Lat, value.Long] for idx , value in Globalcases_df.iterrows()]

LatMean =  Globalcases_df.Lat.mean()

LongMean = Globalcases_df.Long.mean()



# Build features array for use with GeoJson

features = []

for idx, value in Globalcases_df.iterrows():



    time = str(value.Date)



    feature = {

            'type': 'Feature',

            'geometry': {

                'type':'Point', 

                'coordinates':[value.Long,value.Lat]  #GeoJson Coordinates are Longitude, Latitude, Altitude!

            },

            'properties': {

                'time': time,

                'icon': 'circle',

                'iconstyle':{

                    'fillOpacity': 0.8,

                    'color': 'red',

                    'stroke': 'true',

                    'radius': 3,

                   

                }

            }

        }

    features.append(feature)



VirusMap = folium.Map(location=[LatMean , LongMean ], tiles='StamenTerrain', zoom_start=2.5)

            

        

TimestampedGeoJson(

        {'type': 'FeatureCollection',

        'features': features}

        , period='P1D'

        , add_last_point=False

        , auto_play=True

        , loop=True

        , max_speed=1

        , loop_button=True

        , date_options='YYYY-MM-DD'

        , time_slider_drag_update=True

        

    ).add_to(VirusMap)





VirusMap.save('PandemicSpread.html')



HTML(filename = "PandemicSpread.html")

                            

        

    


