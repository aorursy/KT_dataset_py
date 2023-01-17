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
#import libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#load just the state csv

states_filepath = '/kaggle/input/zecon/State_time_series.csv' 

states_data= pd.read_csv(states_filepath)
#view the dataframe

states_data.head()

states_data.tail()



#see all columns in dataframe

print(states_data.columns.values)
states_data.dtypes



#convert date column to date format

states_data['Date'] = pd.to_datetime(states_data['Date'])

states_data['Year']= states_data['Date'].dt.year



#find the year with the most listing

popular_year= states_data['Year'].mode()[0]

print('The year with the most listing is:', popular_year)
#average home value for all homes, by year 

states_data.groupby(['Year']).ZHVIPerSqft_AllHomes.agg([len, min, max])

states_data.groupby(['Year']).ZHVIPerSqft_AllHomes.mean().plot(kind= 'bar', figsize=(20,8))

plt.title("Average value of all homes per square foot, by year", fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.ylabel('Average home value, in $', fontsize=12)
#average of the median listing price, by year

states_data.groupby(['Year']).MedianListingPrice_AllHomes.mean().plot(kind='line', figsize=(20,6))

plt.title('Median listing price of all homes, by Year', fontsize= 14)

plt.xlabel('Year', fontsize= 12)

plt.ylabel('Median listing price, in $', fontsize= 12)
#Average of the median rental price, by year

states_data.groupby(['Year']).MedianRentalPrice_AllHomes.mean().plot(kind='line', figsize=(20,6))

plt.title('Median rental price of all homes, by Year', fontsize= 14)

plt.xlabel('Year', fontsize= 12)



plt.ylabel('Median rental price, in $', fontsize= 12)
#Average of the Median listing price for all years, by state

states_data.groupby(['RegionName']).MedianListingPrice_AllHomes.mean().sort_values(ascending=True).plot(kind='bar', figsize=(22,8))

plt.title('Median listing price of all homes, by State', fontsize= 14)

plt.xlabel('State', fontsize= 12)

plt.ylabel('Median listing price, in $', fontsize= 12)
#Average of the Median rental price for all years, by state

states_data.groupby(['RegionName']).MedianRentalPrice_AllHomes.mean().sort_values(ascending = True).plot(kind='bar', figsize=(22,8))

plt.title('Median rental price of all homes, by State', fontsize= 14)

plt.xlabel('State', fontsize= 12)

plt.ylabel('Median rental price, in $', fontsize= 12)
#average of the median listing price per square, by year

states_data.groupby(['Year']).MedianListingPricePerSqft_AllHomes.mean().plot(kind='line', figsize=(20,6))

plt.title('Median listing price of all homes per square feet, by Year', fontsize= 14)

plt.xlabel('Year', fontsize= 12)

plt.ylabel('Median listing price per square feet, in $', fontsize= 12)
#Average of the median rental price per square feet, by year

states_data.groupby(['Year']).MedianRentalPricePerSqft_AllHomes.mean().plot(kind='line', figsize=(20,6))

plt.title('Median rental price per square feet of all homes, by Year', fontsize= 14)

plt.xlabel('Year', fontsize= 12)

plt.ylabel('Median rental price per square feet, in $', fontsize= 12)
#Average of the Median listing price for all years, by state

states_data.groupby(['RegionName']).MedianListingPricePerSqft_AllHomes.mean().sort_values(ascending=True).plot(kind='bar', figsize=(22,8))

plt.title('Median listing price per squre feet of all homes, by State', fontsize= 14)

plt.xlabel('State', fontsize= 12)

plt.ylabel('Median listing price per square feet, in $', fontsize= 12)
#Average of the Median rental price per square feet for all years, by state

states_data.groupby(['RegionName']).MedianRentalPricePerSqft_AllHomes.mean().sort_values(ascending = True).plot(kind='bar', figsize=(22,8))

plt.title('Median rental price per square feet of all homes, by State', fontsize= 14)

plt.xlabel('State', fontsize= 12)

plt.ylabel('Median rental price per square feet, in $', fontsize= 12)
#How long to sell in days, by year

states_data.groupby(['Year']).DaysOnZillow_AllHomes.mean().plot(kind='line', figsize=(15,7))

plt.title('Days listed on Zillow for all homes, by Year', fontsize= 14)

plt.xlabel('Year', fontsize= 12)

plt.ylabel('Days listed on Zillow', fontsize= 12)
#How long to sell in days, by state

states_data.groupby(['RegionName']).DaysOnZillow_AllHomes.mean().sort_values(ascending= True).plot(kind= 'bar', figsize=(20,6))

plt.title('Days listed on Zillow for all homes, by State', fontsize= 14)

plt.xlabel('State', fontsize= 12)

plt.ylabel('Days listed on Zillow', fontsize= 12)