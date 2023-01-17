# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# URL's for the dynamic data from the multiple sources combined together



confirmed_data_source = '/kaggle/input/covid19-csea/time_series_covid_19_confirmed.csv'

deaths_data_source    = '/kaggle/input/covid19-csea/time_series_covid_19_deaths.csv'

recovered_data_source = '/kaggle/input/covid19-csea/time_series_covid_19_recovered.csv'
# importing various libraries 



import folium        

import numpy as np    

import pandas as pd

from folium.plugins import MarkerCluster
# Reading whole data from csv files



daily_confirmed_data = pd.read_csv(confirmed_data_source)

daily_death_data     = pd.read_csv(deaths_data_source)

daily_recovered_data = pd.read_csv(recovered_data_source)
# Extracting required data i.e. current or latest data



def get_latest_data(df):

  return df.iloc[:, [0, 1, 2, 3, -1]]
def get_map(df):

  df = get_latest_data(df)





  # Function that gives the colour to on the basis of severity of situations



  def color_change(c):

    if(c > 50):

        return('red')

    elif(25 <= c <= 49):

        return('orange')

    elif(10 <= c <= 25):

        return('green')

    else:

        return('yellow')



  # Helper function

  

  def get_province(name):

    if name is np.nan:

      return ''

    else:

      return '(' + name + ')' 

    

  # Create base map

  

  London = [51.506949, -0.122876]

  map = folium.Map(location = London,

                  zoom_start = 2, 

                  tiles = "CartoDB dark_matter")

  

  # Making clusters for better visuals on map



  marker_cluster = MarkerCluster().add_to(map)



  # Adding markers on various locations





  for index, row in df.iterrows(): 

      folium.CircleMarker(location = [row['Lat'], row['Long']],

                          radius = 9, 

                          popup = row['Country/Region'] + get_province(row['Province/State']) + ' ' + str(row[-1]), 

                          fill_color = color_change(row[-1]), 

                          color = "gray", 

                          fill_opacity = 0.9).add_to(marker_cluster)

  return map
get_map(daily_confirmed_data)