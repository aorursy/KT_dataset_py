import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport

import warnings

warnings.filterwarnings("ignore")

mpl.style.use('ggplot') 

%matplotlib inline
data = pd.read_csv('../input/air-quality-data-india-from-20152020/city_day_new.csv')

data.head(5)
data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year
ax = data['AQI'].plot.kde()

ax.set_title('AQI Distibution')
aqi = pd.DataFrame(data.groupby('City')['AQI'].mean())

aqi = aqi.sort_values('AQI')
ax = aqi.plot(kind='barh', 

          figsize = (17, 17), 

          width = 0.4,

          linewidth=2, 

          edgecolor='white')



tt = "Cities with highest AQI"

ax.set_title(tt, fontsize=26)

plt.legend(fontsize = 18)

ax.set_yticklabels(aqi.index.tolist(),fontsize=22)

ax.axes.get_yaxis().set_visible(True)

ax.set_xticklabels([0,100,200,300,400],fontsize=18)

ax.axes.get_xaxis().set_visible(True)

perc = data[['year','City','AQI']]

perc['mean_AQI']=perc.groupby([perc.City,perc.year])['AQI'].transform('mean')

perc.drop('AQI',axis=1,inplace=True)

perc = perc.drop_duplicates().set_index('year')
!jupyter nbextension enable --py --sys-prefix widgetsnbextension
import ipywidgets as widgets

from ipywidgets import HBox, VBox

from IPython.display import display
@widgets.interact_manual(

    city=aqi.index.tolist())

def plot(city='Ahmedabad',grid=True):

    df = perc[perc['City'] == city]

    df.drop('City',axis=1,inplace=True)

    df.plot(kind='bar', figsize = (10, 5), width = 0.1, linewidth=2, edgecolor='white')

    tt = "Year Wise AQI"

    plt.title(tt, fontsize=18)

    plt.xlabel('Year')

    plt.xticks(rotation=0)

    plt.ylabel('AQI')

    plt.legend(fontsize = 16)
city = aqi.index.tolist()

# Removing cities having data of less than 3 years

unwanted_cities = ['Shillong','Ernakulam','Chandigarh','Kochi','Guwahati','Bhopal','Aizawl'] 

for ele in unwanted_cities:

    city.remove(ele)

fig = plt.figure(figsize=(15,28))

for i in range(len(city)):

    df = data[data['City']==city[i]]

    df_year = df.groupby('year')['AQI'].mean().reset_index().dropna()

    ax = fig.add_subplot(6,3,i+1)

    ax.plot(df_year['year'], df_year['AQI'])

    ax.set_xticks(df_year['year'].tolist())

    ax.set_title(city[i])

    ax.set_ylabel('Mean AQI')

fig.tight_layout(pad=0.5)
ele = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',

       'O3', 'Benzene', 'Toluene', 'Xylene']

df = data.groupby('City')[ele].mean()

# Fill the Null values with 0

df = df.fillna(0.0)
@widgets.interact_manual(

    Compound = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',

       'O3', 'Benzene', 'Toluene', 'Xylene'])

def plot(Compound='PM2.5',grid=True):

    curr = df[Compound]

    curr.plot(kind='bar', figsize = (20, 10), width = 0.35, linewidth=2, edgecolor='white')

    tt = "Variation in Compounds"

    plt.title(tt, fontsize=18)

    plt.xlabel('City')

    plt.xticks(rotation=45 , ha='right' ,fontsize=17)

    plt.ylabel('Amount')

    plt.legend(fontsize = 18)
variables = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2','O3', 'Benzene', 'Toluene', 'Xylene']

fig = plt.figure(figsize=(16,34))

for variable,num in zip(variables, range(1,len(variables)+1)):

    ax = fig.add_subplot(6,2,num)

    sns.scatterplot(variable, 'AQI', hue='year', data=data , palette='Oranges')

    plt.title('Relation between {} and AQI'.format(variable))

    plt.xlabel(variable)

    plt.ylabel('AQI')

fig.tight_layout(pad=0.5)
cities = ['Delhi', 'Ahmedabad', 'Patna', 'Gurugram', 'Lucknow']

fig,ax = plt.subplots(figsize=(15, 7))

for city in cities: 

    sns.lineplot(x="Date", y="AQI", data=data[data['City']==city].iloc[::30],label = city)

ax.set_title('AQI values in cities')

ax.legend()