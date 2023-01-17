# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#IMPORTING EXTRAS

import math as math

import matplotlib.pyplot as plt

import random as rand

import datetime as dt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Generation_Plant1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

WeatherSensor_Plant1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
print("The Shape of the Dataframe of Generation of plant 1 is", Generation_Plant1.shape)

print("The Shape of the Dataframe of the Weather Sensor of plant 1 is", WeatherSensor_Plant1.shape)
print("Counts of Generation of plant one")

print(Generation_Plant1.count())

print(" ") #Just for aesthetic

print("Counts of Weather Sensors of plant one")

print(WeatherSensor_Plant1.count())
print("Generation - plant 1")

print(Generation_Plant1.info())

print(" ")

print("Weather Sensor - Plant 1")

print(WeatherSensor_Plant1.info())
print("Generation - plant 1")

print(Generation_Plant1.describe())

print(" ")

print("Weather Sensor - Plant 1")

print(WeatherSensor_Plant1.describe())
print("Generation of Plant 1")

print("First 5")

print(Generation_Plant1.head())

print(" ")

print("Last 5")

print(Generation_Plant1.tail())

print(" ")

print("Weather Sensor Plant 1")

print("First 5")

print(WeatherSensor_Plant1.head())

print(" ")

print("Last 5")

print(WeatherSensor_Plant1.tail())
MeanYield = Generation_Plant1['DAILY_YIELD'].mean()

print("The mean is -",MeanYield)

print(" ")

print("Rounded off -", round(MeanYield, 2))
#Converting from datetime to date

WeatherSensor_Plant1['DATE_TIME'] = pd.to_datetime(WeatherSensor_Plant1['DATE_TIME'], format = '%Y-%m-%d %H:%M')

WeatherSensor_Plant1['DATE'] = pd.to_datetime(WeatherSensor_Plant1['DATE_TIME'], format = '%Y-%m-%d %H:%M').dt.date

WeatherSensor_Plant1['DATE'] = pd.to_datetime(WeatherSensor_Plant1['DATE'], format = '%Y-%m-%d')



#Adding the daily irradiation and adding the values to a list

y = list()

print("No of unique dates -",WeatherSensor_Plant1['DATE'].nunique())   #Number of dates

UniqueDate = WeatherSensor_Plant1['DATE'].unique()     #list of unique dates

for o in UniqueDate:

    Summation = 0                  #Variable reset to make sure each day starts at 0

    for i in range(0,3182):        #since there are 3182 entries

        if WeatherSensor_Plant1['DATE'][i] == o:     #Checks for date match

            Summation = Summation + WeatherSensor_Plant1['IRRADIATION'][i] #The value of irradiation per reading added

    y.append(round(Summation,2))  #X axis

x = list()  #Y axis

y.reverse()

for p in range(0,34):

    x.append(str(UniqueDate[p]))     #converting datetime values to a string

x.reverse()

#Plotting on a bar graph

plt.figure(figsize = (6,10))

plt.barh(x,y, color='green')       

plt.title("Irradiation per day")

plt.xlabel("Irradiation")

plt.ylabel("Dates")

plt.show()
print("Max Ambient Temperature is ", WeatherSensor_Plant1['AMBIENT_TEMPERATURE'].max())

print("Rounded ", round(WeatherSensor_Plant1['AMBIENT_TEMPERATURE'].max(),2))

print(" ")

print("Max Module Temperature is ", WeatherSensor_Plant1['MODULE_TEMPERATURE'].max())

print("Rounded ", round(WeatherSensor_Plant1['MODULE_TEMPERATURE'].max(),2))
#Since each invertor has its own source key, the number of invertors is the number of unique source keys, so

print("Number of Inverters is", Generation_Plant1['SOURCE_KEY'].nunique())
print("DC Power")

print("MAX -", Generation_Plant1['DC_POWER'].max())

print("MIN -", Generation_Plant1['DC_POWER'].min())

print(" ")

print("AC Power")

print("MAX -", Generation_Plant1['AC_POWER'].max())

print("MIN -", Generation_Plant1['AC_POWER'].min())
idSource = Generation_Plant1['DC_POWER'].idxmax()    #Gets row no of the highest DC generated time

#Now we can use this to track down the inverter's source key

print("The inverter's source key is", Generation_Plant1['SOURCE_KEY'][idSource])



#Now we use the same principle for AC

idSource2 = Generation_Plant1['AC_POWER'].idxmax()

print("The inverter's source key is", Generation_Plant1['SOURCE_KEY'][idSource2])
#To rank inverters we must judge them on the sum of DC to the source key

ValsDC = sorted(Generation_Plant1.groupby('SOURCE_KEY')['DC_POWER'].sum(), reverse = 1) #Sorting the Total Yield in reverse order

nnnn = Generation_Plant1['DC_POWER'].count()      #Taking no of values in total yield

UniqueSource = Generation_Plant1['SOURCE_KEY'].unique()

sigma = float()      #Helps in addition

Sums = list()        #Declares list of Sums

for out in range(0,22):

    sigma = 0

    for IN in range(0,nnnn):

        if UniqueSource[out] == Generation_Plant1['SOURCE_KEY'][IN]:     #Checks wether the source key of the cell IN is equal to the OUTth unique source key

            sigma = sigma + Generation_Plant1['DC_POWER'][IN]     #If true, adds

    Sums.append(sigma)    #Appends Sigma once value is finalised

print("The Rank based on DC Power is")

for o in range(0,22):

    for i in range (0,22):

        if ValsDC[o] == Sums[i]:

            print(Generation_Plant1['SOURCE_KEY'][i])     #Since ValsDC is in order, Sums matches with the sum in ValsDC to get the rank
ValsDC = sorted(Generation_Plant1.groupby('SOURCE_KEY')['AC_POWER'].sum(), reverse = 1) 

nnnn = Generation_Plant1['AC_POWER'].count()

UniqueSource = Generation_Plant1['SOURCE_KEY'].unique()

sigma = float()

Sums = list()

for out in range(0,22):

    sigma = 0

    for IN in range(0,nnnn):

        if UniqueSource[out] == Generation_Plant1['SOURCE_KEY'][IN]:

            sigma = sigma + Generation_Plant1['AC_POWER'][IN]

    Sums.append(sigma)

print("The Rank based on AC Power is")

for o in range(0,22):

    for i in range (0,22):

        if ValsDC[o] == Sums[i]:

            print(Generation_Plant1['SOURCE_KEY'][i])
#First Splitting Date and Time

Generation_Plant1['DATE_TIME'] = pd.to_datetime(Generation_Plant1['DATE_TIME'], format = '%d-%m-%Y %H:%M')

Generation_Plant1['DATE'] = pd.to_datetime(Generation_Plant1['DATE_TIME'], format = '%d-%m-%Y %h:%M').dt.date

Generation_Plant1['DATE'] = pd.to_datetime(Generation_Plant1['DATE'], format = '%Y-%m-%d')



#Since a reading is taken every 15 minutes, every 1 hour 4 readings are taken. There are 22 invertors. Therefore everyday 24*4*22 = 2112 readings should be taken.



CountingVar = Generation_Plant1['DATE'].value_counts()

print("Generation in plant 1 (out of 2112)")

print(CountingVar.sort_index())

print(" ")



#A reading every 15 mins => 24x4 readings per day = 96 readings per day in WeatherSensor_Plant1



CountingVar2 = WeatherSensor_Plant1['DATE'].value_counts()

print("Weather Sensor Plant 1 (out of 96)")

print(CountingVar2.sort_index())