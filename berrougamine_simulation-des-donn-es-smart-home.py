from IPython.display import Image

Image(url="https://cdn1.imggmi.com/uploads/2019/11/17/691e8d814b05591d979118a6c0e8c768-full.jpg")
#Importation des bibliothèques de visualisation



import pandas as pd

import numpy as np

from pandas.plotting  import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection 

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

import seaborn as sns

import warnings#ignore alertes

warnings.filterwarnings('ignore')

    
#Importation des données

my_filepath ='../input/smart-home-dataset-with-weather-information/HomeC.csv'

#lecture duy fichier par variable 

my_data = pd.read_csv(my_filepath  ,   parse_dates=True)

home_dat = my_data.select_dtypes(exclude=['object'])

#Indexation du temps

time_index = pd.date_range('2016-01-01 05:00', periods=503911,  freq='min')  

time_index = pd.DatetimeIndex(time_index)

home_dat = home_dat.set_index(time_index)
#Donner des nouveaux noms aux attributs 

energy_data = home_dat.filter(items=[ 'gen [kW]', 'House overall [kW]', 'Dishwasher [kW]',

                                     'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 'Fridge [kW]',

                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'])



weather_data = home_dat.filter(items=['temperature',

                                      'humidity', 'visibility', 'apparentTemperature', 'pressure',

                                      'windSpeed', 'windBearing', 'dewPoint'])
#Visualisation des premiers enregistrements 

energy_data.head()
#données Meteo

weather_data.head()
#la consommation d'énérgie par jour

energy_per_day = energy_data.resample('D').sum()

energy_per_day.head()
#Energie consommé pendat un mois

energy_per_month = energy_data.resample('M').sum() 

plt.figure(figsize=(20,10))

sns.lineplot(data= energy_per_month.filter(items=[ 'Dishwasher [kW]','House overall [kW]',

                                     'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 'Fridge [kW]',

                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]']) , dashes=False  )

# VIsualisation de la consommation des les chambres

sns.lineplot(data= energy_per_month.filter(items=[      

                                     'Home office [kW]',

                                     'Wine cellar [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',

                                      'Living room [kW]']) , dashes=False  )
#la méteo enregistré par jour/ mois

weather_per_day = weather_data.resample('D').mean()  

weather_per_day.head()

weather_per_month = weather_data.resample('M').mean()               
#Visualisation de la méteo

plt.figure(figsize=(20,8))

sns.lineplot(data= weather_per_month.filter(items=['temperature',

                                      'humidity', 'visibility', 'apparentTemperature',

                                      'windSpeed', 'dewPoint']) ,dashes=False )
#Distribution de la temperature

weather_data['temperature'].plot(figsize=(25,5))
#Visualisation de la consommation de la date 01-10-2016

plt.figure(figsize=(20,8))

sns.lineplot(data= energy_data.loc['2016-10-01 00:00' : '2016-10-02 00:00'].filter([ 'Home office [kW]',

                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                 'Living room [kW]']),dashes=False , )