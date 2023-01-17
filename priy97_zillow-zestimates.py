# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from IPython.display import Image, display



import plotly 

import plotly.offline as py

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.figure_factory as fig_fact 





%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# loading required data into dataframes



df_city_time_series = pd.read_csv('/kaggle/input/zecon/City_time_series.csv')



#head of the data



list(df_city_time_series.columns)





df_cities_crosswalk = pd.read_csv('/kaggle/input/zecon/cities_crosswalk.csv')



df_cities_crosswalk.head()
#barplot



df_city_time_series.Date = pd.to_datetime(df_city_time_series.Date)



df_city_time_series.groupby(df_city_time_series.Date.dt.year)['ZHVIPerSqft_AllHomes'].mean().plot(kind='bar')



plt.title('Median of the value of all homes per sq.ft yearly', fontsize=11)

plt.ylabel('Home Value per Sqft')

plt.xlabel('Year')

plt.show()
#drop the null values

df_city_time_series_without_null = df_city_time_series.dropna(subset=['MedianListingPricePerSqft_AllHomes'])



df_city_time_series_without_null.groupby(df_city_time_series_without_null.Date.dt.year)['MedianListingPricePerSqft_AllHomes'].mean().plot(kind='bar')



plt.title('Median List Price per Sqft Yearly', fontsize=24)



plt.xlabel('Year')

plt.ylabel('Median List Price per Sqft ')

##  GRAPH INCORRECT - FIX LATER

#drop the null values

df_city_time_series_without_null_rent = df_city_time_series.dropna(subset=['MedianRentalPricePerSqft_AllHomes'])



df_city_time_series_without_null_rent.groupby(df_city_time_series_without_null_rent.Date.dt.year)['MedianRentalPricePerSqft_AllHomes'].mean().plot(kind='bar', figsize=(10, 6))



plt.title('Median Rental Price per Sqft Yearly', fontsize=24)





plt.xlabel('Year')

plt.ylabel('Median Rental Price per Sqft ')
df_city_time_series.groupby(df_city_time_series.Date.dt.year)[['ZHVI_2bedroom','ZHVI_3bedroom', 'ZHVI_4bedroom']].mean().plot(kind='bar', figsize=(10, 6))



plt.title('Zillow Home Values Annually')

plt.xlabel('Year')

plt.ylabel('Zillow Home Value')
df_city_time_series_without_null_sold = df_city_time_series.dropna(subset=['Sale_Counts'])



df_city_time_series_without_null_sold.groupby(df_city_time_series_without_null_sold.Date.dt.year)['Sale_Counts'].mean().plot(kind='bar')