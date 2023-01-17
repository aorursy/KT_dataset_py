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
#FIRST WE SHALL CONSIDER SOLAR POWER GENERATION DATA AND WEATHER SENSOR DATA OF PLANT 1 
df_plant1spgd = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
#Loading and Reading The CSV file of Solar Power Generation Data for Plant 1
df_plant1wsd = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
#Loading and Reading The CSV file of Weather Sensor Data for Plant 1
#Question 1: What is the mean value of daily yield of Plant 1?


df_plant1spgd['DAILY_YIELD'] 
#To view the Daily Yield Column of Plant 1
df_plant1spgd['DAILY_YIELD'].mean() 
#Gives you the mean value of daily yield of Plant 1
#Question 2: What is the total irradiation per day of Plant 1?

df_plant1wsd['DATE_TIME'] = pd.to_datetime(df_plant1wsd['DATE_TIME'])   
#Code for changing to date time
df_plant1wsd['year'] = df_plant1wsd['DATE_TIME'].dt.year
df_plant1wsd['month'] = df_plant1wsd['DATE_TIME'].dt.month
df_plant1wsd['day'] = df_plant1wsd['DATE_TIME'].dt.day
df_plant1wsd['date'] = df_plant1wsd['DATE_TIME'].dt.date
#Creating new Columns for year, month, day and date
df_plant1wsd 
#To view the Weather Sensor Data of Plant 1
IRR1 = df_plant1wsd.groupby(['date']).sum()
#Grouping the data by date and Calculating the sum of Irridation per day for Plant 1
IRR1["IRRADIATION"]
#Gives you the amount of Irridation per day for Plant 1
#Question 3: What is the max ambient and module temperature of Plant 1?

df_plant1wsd[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']] 
#To view the Ambient Temeprature and Module Temperature Columns of Plant 1
df_plant1wsd[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max() 
#Gives you the maximum values of the Ambient Temperature and Module Temperature of Plant 1
#Question 4: How many inverters are there for of Plant 1? 
df_plant1spgd['SOURCE_KEY'] 
#To view the Source Key Column of Plant 1
df_plant1spgd['SOURCE_KEY'].unique()
#Gives the Source Keys of all the inverters of Plant 1
len(df_plant1spgd['SOURCE_KEY'].unique())
#The Number of inverters of Plant 1
#Question 5: What is the maximum/minimum amount of DC/AC Power generated in a time interval/day for Plant 1?
df_plant1spgd['DATE_TIME'] = pd.to_datetime(df_plant1spgd['DATE_TIME'])   
#Code for changing to date time
df_plant1spgd['year'] = df_plant1spgd['DATE_TIME'].dt.year
df_plant1spgd['month'] = df_plant1spgd['DATE_TIME'].dt.month
df_plant1spgd['day'] = df_plant1spgd['DATE_TIME'].dt.day
df_plant1spgd['date'] = df_plant1spgd['DATE_TIME'].dt.date
#Creating new Columns for year, month, day and date
df_plant1spgd
#To view the Solar Power Generation Data for Plant 1
DCMAX1 = df_plant1spgd.groupby(['date']).max()
#Grouping the data by date and Calculating the Maximum DC Power reading of the day for Plant 1
DCMAX1["DC_POWER"]
#Gives you the Maximum DC Power reading of the day for Plant 1
DCMIN1 = df_plant1spgd.groupby(['date']).min()
#Grouping the data by date and Calculating the Minimum DC Power reading of the day for Plant 1
DCMIN1["DC_POWER"]
#Gives you the Minimum DC Power reading of the day for Plant 1
ACMAX1 = df_plant1spgd.groupby(['date']).max()
#Grouping the data by date and Calculating the Maximum AC Power reading of the day for Plant 1
ACMAX1["AC_POWER"]
#Gives you the Maximum AC Power reading of the day for Plant 1
ACMIN1 = df_plant1spgd.groupby(['date']).min()
#Grouping the data by date and Calculating the Minimum AC Power reading of the day for Plant 1
ACMIN1["AC_POWER"]
#Gives you the Minimum AC Power reading of the day for Plant 1
#Question 6: Which inverter (source_key) has produced maximum DC/AC power in Plant 1?

SKDCMAX1 =  df_plant1spgd[['SOURCE_KEY','DC_POWER']][df_plant1spgd.DC_POWER == (df_plant1spgd.DC_POWER).max()]
#Code to display the Source Key and DC Power Columns and Finding the inverter having the Maximum DC Power reading in these 34 days for Plant 1
print(SKDCMAX1)
#Displays the Source Key of the Inverter which produced the Maximum DC Power Reading in Plant 1
SKACMAX1 = df_plant1spgd[['SOURCE_KEY','AC_POWER']][df_plant1spgd.AC_POWER == df_plant1spgd.AC_POWER].max()
#Code to display the Source Key and AC Power Columns and Finding the inverter having the Maximum AC Power reading in these 34 days for Plant 1
print(SKACMAX1)
#Displays the Source Key of the Inverter which produced the Maximum AC Power Reading in Plant 1
#Question 7: Rank the inverters based on the DC/AC power they produce in Plant 1?
SKRANK1 = df_plant1spgd.groupby(['SOURCE_KEY']).count()
#Code for grouping the data based on the Source Key and using the count function to count the number of inverters in Plant 1

SKRANK1.iloc[:,[2,3]].rank()
#Code for ranking the inverters based on the production of DC and AC Power they produce for Plant 1
#Question 8: Is there any missing data in Plant 1?

df_plant1spgd.isnull()
#To find whether there is any missing data in given with respect to Solar Power Generation for Plant 1
#Here is another way to check whether there is any missing data in the Solar Power Generation Data
df_plant1spgd.notnull()

df_plant1wsd.isnull()
#To find whether there is any missing data in given with respect to Sensing the Weather for Plant 1
#Here is another way to check whether there is any missing data in the Weather Sensor Data
df_plant1wsd.notnull()
#BUT ONE THING IS FOR SURE. THE AMOUNT OF DATA IS HUGE. 
#JUST BECAUSE IT IS NOT VISIBLE TO US NOW, DOESEN'T MEAN THAT THERE IS NO MISSING DATA AT ALL
#THERE IS MORE THAN WHAT MEETS THE EYE
#NOW WE SHALL CONSIDER SOLAR POWER GENERATION DATA AND WEATHER SENSOR DATA OF PLANT 2
df_plant2spgd = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
#Loading and Reading The CSV file of Solar Power Generation Data for Plant 2
df_plant2wsd = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
#Loading and Reading The CSV file of Weather Sensor Data for Plant 2
#Question 1: What is the mean value of daily yield of Plant 2?
df_plant2spgd['DAILY_YIELD'] 
#To view the Daily Yield Column of Plant 2
df_plant2spgd['DAILY_YIELD'].mean() 
#Gives you the mean value of daily yield of Plant 2
#Question 2: What is the total irradiation per day of Plant 2?

df_plant2wsd['DATE_TIME'] = pd.to_datetime(df_plant2wsd['DATE_TIME'])   
#Code for changing to date time
df_plant2wsd['year'] = df_plant2wsd['DATE_TIME'].dt.year
df_plant2wsd['month'] = df_plant2wsd['DATE_TIME'].dt.month
df_plant2wsd['day'] = df_plant2wsd['DATE_TIME'].dt.day
df_plant2wsd['date'] = df_plant2wsd['DATE_TIME'].dt.date
#Creating new Columns for year, month, day and date
df_plant2wsd 
#To view the Weather Sensor Data of Plant 2
IRR2 = df_plant2wsd.groupby(['date']).sum()
#Grouping the data by date and Calculating the sum of Irridation per day for Plant 2
IRR2["IRRADIATION"]
#Gives you the amount of Irridation per day for Plant 2
#Question 3: What is the max ambient and module temperature of Plant 2?
df_plant2wsd[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']] 
#To view the Ambient Temeprature and Module Temperature Columns of Plant 2
df_plant2wsd[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max() 
#Gives you the maximum values of the Ambient Temperature and Module Temperature of Plant 2
#Question 4: How many inverters are there for of Plant 2? 
df_plant2spgd['SOURCE_KEY'] 
#To view the Source Key Column of Plant 2
df_plant2spgd['SOURCE_KEY'].unique()
#Gives the Source Keys of all the inverters of Plant 2
len(df_plant2spgd['SOURCE_KEY'].unique())
#The Number of inverters of Plant 2
#Question 5: What is the maximum/minimum amount of DC/AC Power generated in a time interval/day for Plant 2?
df_plant2spgd['DATE_TIME'] = pd.to_datetime(df_plant2spgd['DATE_TIME'])   
#Code for changing to date time
df_plant2spgd['year'] = df_plant2spgd['DATE_TIME'].dt.year
df_plant2spgd['month'] = df_plant2spgd['DATE_TIME'].dt.month
df_plant2spgd['day'] = df_plant2spgd['DATE_TIME'].dt.day
df_plant2spgd['date'] = df_plant2spgd['DATE_TIME'].dt.date
#Creating new Columns for year, month, day and date
df_plant2spgd
#To view the Solar Power Generation Data for Plant 2
DCMAX2 = df_plant2spgd.groupby(['date']).max()
#Grouping the data by date and Calculating the Maximum DC Power reading of the day for Plant 2
DCMAX2["DC_POWER"]
#Gives you the Maximum DC Power reading of the day for Plant 2
DCMIN2 = df_plant1spgd.groupby(['date']).min()
#Grouping the data by date and Calculating the Minimum DC Power reading of the day for Plant 2
DCMIN2["DC_POWER"]
#Gives you the Minimum DC Power reading of the day for Plant 2
ACMAX2 = df_plant2spgd.groupby(['date']).max()
#Grouping the data by date and Calculating the Maximum AC Power reading of the day for Plant 2
ACMAX1["AC_POWER"]
#Gives you the Maximum AC Power reading of the day for Plant 2
ACMIN2 = df_plant2spgd.groupby(['date']).min()
#Grouping the data by date and Calculating the Minimum AC Power reading of the day for Plant 2
ACMAX2["AC_POWER"]
#Gives you the Minimum AC Power reading of the day for Plant 2
#Question 6: Which inverter (source_key) has produced maximum DC/AC power in Plant 1?
SKDCMAX2 =  df_plant2spgd[['SOURCE_KEY','DC_POWER']][df_plant2spgd.DC_POWER == df_plant2spgd.DC_POWER].max()
#Code to display the Source Key and DC Power Columns and Finding the inverter having the Maximum DC Power reading in these 34 days for Plant 2
print(SKDCMAX2)
#Displays the Source Key of the Inverter which produced the Maximum DC Power Reading in Plant 2
SKACMAX2 =  df_plant2spgd[['SOURCE_KEY','AC_POWER']][df_plant2spgd.AC_POWER == df_plant2spgd.AC_POWER].max()
#Code to display the Source Key and AC Power Columns and Finding the inverter having the Maximum AC Power reading in these 34 days for Plant 2
print(SKACMAX2)
#Displays the Source Key of the Inverter which produced the Maximum AC Power Reading in Plant 2

#Question 7: Rank the inverters based on the DC/AC power they produce in Plant 2
SKRANK2 = df_plant2spgd.groupby(['SOURCE_KEY']).count()
#Code for grouping the data based on the Source Key and using the count function to count the number of inverters in Plant 2

SKRANK2.iloc[:,[2,3]].rank()
#Code for ranking the inverters based on the production of DC and AC Power they produce for Plant 2
#Question 8: Is there any missing data in Plant 2?
df_plant2spgd.isnull()
#To find whether there is any missing data in given with respect to Solar Power Generation for Plant 2
df_plant2wsd.isnull()
#To find whether there is any missing data in given with respect to Sensing the Weather for Plant 2
#BUT ONE THING IS FOR SURE. THE AMOUNT OF DATA IS HUGE. 
#JUST BECAUSE IT IS NOT VISIBLE TO US NOW, DOESEN'T MEAN THAT THERE IS NO MISSING DATA AT ALL
#THERE IS MORE THAN WHAT MEETS THE EYE