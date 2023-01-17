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
from datetime import datetime

import pandas as pd

MVC = pd.read_csv("../input/motor-vehicle-collisions-crashes-nyc/Motor_Vehicle_Collisions_-_Crashes.csv")

import pandas as pd

WE = pd.read_csv("../input/weather-events-20162019/WeatherEvents_Aug16_Dec19_Publish.csv")
import pandas as pd

j = pd.read_csv("../input/jfk-airport-station-weather-20162019/2054777.csv")
crash_area_df=MVC["BOROUGH"].value_counts() #count number of crashes in each borough

crash_area_df.plot(kind="bar") #create a bar graph
crash_factor_df=MVC["CONTRIBUTING FACTOR VEHICLE 1"].value_counts()#count number of crashes for each factor

crash_factor_df.head(20).plot(kind="bar")

MVC = pd.read_csv("../input/motor-vehicle-collisions-crashes-nyc/Motor_Vehicle_Collisions_-_Crashes.csv",parse_dates=['CRASH DATE'])

MVC['CRASH DATE']=pd.to_datetime(MVC['CRASH DATE'])#reformat the date to match the format in the other dataset

crash_date_df=MVC["CRASH DATE"].value_counts()

crash_date_df.head(10)
crash_date_df.head(10).plot(kind="bar")
crash_date_df.to_csv('file2.csv')#write into a csv file

crash_date_df = pd.read_csv("file2.csv", names=['DATE', '# of Crashes'])#add headers

crash_date_df.to_csv('file2.csv')#write into a csv file

crash_date_df

NEW = pd.read_csv("file2.csv")

NEW.head()

result = pd.merge(NEW,

                 j[['DATE', 'SNOW', 'PRCP']],

                 on='DATE') #merge new data with weather data, matching the date

result.head(20)
result.head(20).plot(kind="bar",x='# of Crashes',y=["PRCP","SNOW"])

#df.plot(kind='scatter',x='num_children',y='num_pets',color='red')
