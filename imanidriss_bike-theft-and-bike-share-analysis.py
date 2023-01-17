# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import geopandas as gpd

import matplotlib.pyplot as plt

from numpy import arange



import datetime as dt

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_bicycle_thefts = gpd.read_file("../input/tps-toronto-bicycle-thefts/TPS_Toronto_Bicycle_Thefts.geojson")
df_bicycle_thefts.info()

df_bicycle_thefts.head()
df_bicycle_thefts.describe()
theft_location_value = df_bicycle_thefts["Location_Type"].value_counts().iloc[0:15]

theft_location = df_bicycle_thefts["Location_Type"].value_counts().iloc[0:15].index



fig = px.bar(df_bicycle_thefts, x= theft_location , y=theft_location_value)
fig.update_layout(

    title='Types of locations with highest number of thefts',

    xaxis_title = 'Location types',

    yaxis_title ='Number of thefts',

    xaxis_type = 'category'

)





fig.update_traces(marker_color='purple')



fig.show()
list = [

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv',

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv',

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv',

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv'

       ]







for i in list:  #looped through the list

    #print(i)

    

    ridership = pd.read_csv(i) #read the filepath

    #print(ridership['trip_duration_seconds'].isnull().values.any()) #to check if any nun/missing values exist

    

    #print(ridership.head()) #to explore the first five rows

    #print(ridership.tail()) #to explore the last five rows

    #print(ridership.describe()) #to explore the summary statistics
def convert_seconds_to_minutes(seconds):

  minutes = int(seconds / 60);

  seconds = seconds % 60;

  return f"{minutes}"





#return f"{minutes}{seconds}" to also show the seconds


list = [    

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv',

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv',

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv',

    '../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv'

       ]



trip_duration_item = [] #created a list of data sets





for i in list:  #looped through the list

    #print(i)

    

    ridership = pd.read_csv(i) #read the filepath

    trip_duration_string = [] #create a list for strings

    

    for seconds in ridership['trip_duration_seconds']: #gets the seconds in trip duration

       trip_duration_string.append(convert_seconds_to_minutes(seconds)) #converts it to minutes

    

    

    p = pd.Series(trip_duration_string) #take our new list and make it list

    #print (p)



    trip_duration_item.append(p) #appended it back to our first list trip_duration_item

    #print(trip_duration_item[0])
year = ['  2017 Q1', '   2017 Q2', '   2017 Q3', '   2017 Q4', '   2018 Q1', '   2018 Q2', '   2018 Q3', '   2018 Q4']

people_num = [132123, 333353, 663488, 363405, 178559, 558370, 822536, 363490]

#ax = plt.plot



plt.plot(year, people_num, c='DarkGoldenRod')



plt.xticks(rotation = 60) 

plt.tick_params(color = 'DarkGoldenRod', labelsize = '11', pad=20)





#plt.spines["right"].set_visible(False)

    

font = {'fontsize': 20,

 'color' : 'Black',

 }



plt.title('Number of people that use bike-share in different times/quarters', fontdict= font)

plt.xlabel('PERIOD')

plt.ylabel('RIDERSHIP')



plt.show()
def draw_histogram(trip_duration_item, color, range_max, bin_num, hist_title): #creating a function to create histograms



    col = [color]

    x_label = {'x':'Duration(in minutes)'}

    x_range = [0, range_max]



    fig = px.histogram(x=trip_duration_item,

                       labels = x_label,

                       color_discrete_sequence= col,

                       marginal= 'box',

                       barmode = 'overlay',

                       range_x= x_range,

                       nbins=bin_num,

                       title= hist_title,

                       width=800, 

                       height=500)



    fig.show()
#draw_histogram(trip_duration_item[0],'SkyBlue', 30, 700, 'Trip duration of people who used bike-share(Jan-Mar 2017)')

#draw_histogram(trip_duration_item[1],'Pink', 30, 700,'Trip duration of people who used bike-share(Apr-Jun 2017)')

#draw_histogram(trip_duration_item[2],'Plum', 30, 160000,'Trip duration of people who used bike-share(Jul-Sep 2017)')

#draw_histogram(trip_duration_item[3],'PaleVioletRed', 30, 40000, 'Trip duration of people who used bike-share(Oct-Dec 2017)')
draw_histogram(trip_duration_item[0],'Teal', 30, 5000, 'Trip duration of people who used bike-share(Jan-Mar 2018)')
draw_histogram(trip_duration_item[1],'MediumVioletRed', 30, 5000,'Trip duration of people who used bike-share(Apr-Jun 2018)')
draw_histogram(trip_duration_item[2],'DarkOrange', 30, 5000,'Trip duration of people who used bike-share(Jul-Sep 2018)')
draw_histogram(trip_duration_item[3],'SteelBlue', 30, 6400, 'Trip duration of people who used bike-share(Oct-Dec 2018)')