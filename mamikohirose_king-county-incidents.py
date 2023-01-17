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
import pandas as pd

king = pd.read_csv("../input/King_County_Sheriff_s_Office_-_Incident_Dataset.csv") #define dataframe name
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt



import altair as alt #imports altair

from altair import Chart, X, Y, Axis, SortField  

king.sort_values('incident_datetime')

king.head() #good to understand what fields are available.

print(king.shape) #good to understand how many rows there are
king['incident_datetime']= pd.to_datetime(king['incident_datetime']) #cleaning the incident_date_time field

king['date'] = king['incident_datetime'].dt.date #parsing out the date

king['month_year'] = pd.to_datetime(king['date']).dt.to_period('M') #parsing out the month

king['year'] = pd.to_datetime(king['date']).dt.to_period('Y') #parsing out the year

king.head()
 #picked out the columns I'm interested in 



king1 = king[['date','year','month_year','incident_type','city','hour_of_day','day_of_week']]

king1.head()
king1.incident_type.value_counts().plot(kind='bar')
king1.year.value_counts().plot(kind='bar')
#filtered to just 2019 and 2018 based on chart above 

king_filtered = king1[king1['year'] >= 2018] 

print(king_filtered.shape)

print(king1.shape)
king_filtered2 = king_filtered[king_filtered['incident_type'] =='Other']
king_filtered2.hour_of_day.value_counts().plot(kind='bar')
king_filtered2.day_of_week.value_counts().plot(kind='bar')
king_filtered3 = king_filtered[king_filtered['incident_type'] =='Property Crime']

king_filtered3.hour_of_day.value_counts().plot(kind='bar')