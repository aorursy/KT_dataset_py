# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import rcParams

import missingno as msno

plt.style.use('seaborn-whitegrid')

# Let's ignore warnings for now

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data and verifying it



df=pd.read_csv('../input/daily-temperature-of-major-cities/city_temperature.csv')

df.head()
# Check for missing values



df.isnull().sum()
# Seeing the unique values of Year we see that there are a few values where Year=201 and 200. We will drop them



# Remove these records

df=df.drop(df[(df['Year']==201) | (df['Year']==200)].index)

df['Year'].value_counts()
## Seeing the movement of average mean temperature over the years for the entire planet



pd.pivot_table(df,index=['Year'],values=['AvgTemperature'],aggfunc=np.mean).plot(kind='line',color='orange')

plt.title("Average Temperature variation over the years", loc='center', fontsize=12, fontweight=0, color='orange')

plt.xlabel("Year")

plt.ylabel("Average Temperature")


regions = ['North America', 'Europe', 'Asia', 'Africa', 'South/Central America & Carribean', 'Middle East', 'Australia/South Pacific']



# Loop through each region and plot average temperature

plt.figure(figsize=(15,7.5))

for region in regions:

    temp = df[df['Region']== region]

    temp = temp.groupby("Year")["AvgTemperature"].mean()

    a = temp

    plt.plot(a)

    

plt.legend(regions, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.title('Average Annual Temperature by Region')

plt.xlabel('Year')

plt.ylabel('Average Temperature, Fahrenheit')
## Seeing the regions with the most temperature rises from 1995 to 2020



avg_temperature={}

keys=df['Region'].unique()

for region in df.Region.unique():

    avg_temperature[region]=df[(df['Region']==region)&(df['Year']==2020)]['AvgTemperature'].mean()-df[(df['Region']==region)&(df['Year']==1995)]['AvgTemperature'].mean()

avg_temperature_rise=pd.DataFrame(list(avg_temperature.items()),columns=['Region','Average Temperature rise'])



sns.barplot(x='Average Temperature rise',y='Region',data=avg_temperature_rise.sort_values(by='Average Temperature rise',ascending=False)).set_title('Increase in Average Temperature by Region')
## Seeing the countries with the highest and lowest temperature rises from 1995 to 2020



avg_temperature_country={}

keys=df['Country'].unique()

for country in df.Country.unique():

    avg_temperature_country[country]=df[(df['Country']==country)&(df['Year']==2020)]['AvgTemperature'].mean()-df[(df['Country']==country)&(df['Year']==1995)]['AvgTemperature'].mean()

avg_temperature_rise_co=pd.DataFrame(list(avg_temperature_country.items()),columns=['Country','Average Temperature rise'])





fig, ax =plt.subplots(1,2,figsize=(24, 6))

sns.barplot(x='Average Temperature rise',y='Country',ax=ax[0],data=avg_temperature_rise_co.sort_values(by='Average Temperature rise',ascending=False).head(10)).set_title('Top 10 countries with most increase in Average Temperature')

sns.barplot(x='Average Temperature rise',y='Country',ax=ax[1],data=avg_temperature_rise_co.sort_values(by='Average Temperature rise',ascending=True).head(10)).set_title('Top 10 countries with least increase in Average Temperature')
## Seeing the cities with the highest and lowest temperature rises from 1995 to 2020



avg_temperature_city={}

keys=df['City'].unique()

for city in df.City.unique():

    avg_temperature_city[city]=df[(df['City']==city)&(df['Year']==2020)]['AvgTemperature'].mean()-df[(df['City']==city)&(df['Year']==1995)]['AvgTemperature'].mean()

avg_temperature_rise_ci=pd.DataFrame(list(avg_temperature_city.items()),columns=['City','Average Temperature rise'])





fig, ax =plt.subplots(1,2,figsize=(24, 6))

sns.barplot(x='Average Temperature rise',y='City',ax=ax[0],data=avg_temperature_rise_ci.sort_values(by='Average Temperature rise',ascending=False).head(10)).set_title('Top 10 cities with most increase in Average Temperature')

sns.barplot(x='Average Temperature rise',y='City',ax=ax[1],data=avg_temperature_rise_ci.sort_values(by='Average Temperature rise',ascending=True).head(10)).set_title('Top 10 cities with least increase in Average Temperature')
## Isolating India data set



df_india = df[df.Country == 'India'].copy()

df_india.head()
plt.figure(figsize=(20,8))

sns.lineplot(x = 'Year', y = 'AvgTemperature', data = df_india , palette='hsv')

plt.title('Average Temperatures in India')

plt.ylabel('Average Temperature')

plt.xlabel('')

plt.xticks(range(1995,2020))

plt.show()
## Seeing the monthly variation of temperature in India over time



months = ['January', 'February', 'March', 'April', 'May', 'June', 

          'July', 'August', 'September', 'October', 'November', 'December']
## Seeing the monthly variation of temperature in India over time

fig = plt.subplots(3,4, figsize = (15,8))

for i in range(1,13): 

    ax = plt.subplot(3, 4, i)

    sns.lineplot(x = 'Day', y = 'AvgTemperature', data = df_india[df_india.Month == i] , palette='hsv')

    ax.title.set_text(months[i-1])

    #ax.set_ylim((-5,25))

    ax.set_xlabel('')

    ax.set_ylabel('')

plt.suptitle('Monthly Temperatures in India (1995-2019)', y = 1.05)

#plt.ylabel('Average Temperature (°C)')

plt.tight_layout()

plt.show();
## Monthly average temperature in India



india_pivoted = pd.pivot_table(data= df_india,index='Month',values='AvgTemperature',columns='Year')

plt.figure(figsize=(20, 8))

sns.heatmap(data = india_pivoted, cmap='coolwarm', annot = True)

plt.ylabel('Month')

plt.xlabel('')

plt.title('Average Temperatures in India')

plt.show();