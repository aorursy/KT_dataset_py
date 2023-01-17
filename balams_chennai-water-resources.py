#Load Libraries

import numpy as np

import pandas as pd



#EDA

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')



#Read Water Level

water_level = pd.read_csv('../input/chennai_reservoir_levels.csv')

water_level.head(3)
#Basic Info

basic = {'unique' : water_level.nunique(),

        'null_value' : water_level.isna().sum(),

        'data_type' : water_level.dtypes}



print('Shape :',water_level.shape)

pd.DataFrame(basic)
water_level.describe().transpose()
#Convert Date Column into datetime data type

water_level.Date = pd.to_datetime(water_level.Date)

water_level.dtypes
#Add Three more columns (Year, Month, quarter,, Total)

water_level['Year'] = water_level.Date.dt.year

water_level['Month'] = water_level.Date.dt.month

water_level['Total'] = water_level.POONDI + water_level.CHOLAVARAM + water_level.REDHILLS + water_level.CHEMBARAMBAKKAM



water_level.Month.replace([1,2,3,4,5,6,7,8,9,10,11,12],

             ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], inplace = True)



water_level.Month = pd.Categorical(water_level.Month,

                                   ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 

                                   ordered = True )







lakes = ['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM','Total']

water_level.head(3)
#EDA - Year

year = water_level.groupby('Year').sum().reset_index()



plt.figure(figsize = (15,35))



i = 0

for lake in lakes:

    plt.subplot(6,2, i+1)

    sns.barplot( x= 'Year' ,y = lake, data = year)

    plt.xticks(rotation = 45)

    plt.xlabel('Year')

    plt.ylabel(lake)

    plt.title(lake + ' vs Year', size = 10)

        

    plt.subplot(6,2, i+2)

    sns.lineplot(x = 'Year', y = lake, data = water_level,ci = False)

    plt.xticks(rotation = 45)

    plt.xlabel('Year')

    plt.ylabel(lake)

    plt.title(lake + ' vs Year', size = 10)

    plt.subplots_adjust(hspace = 0.5)

    i+=2



plt.show()
#EDA - Month

month = water_level.groupby('Month').mean().reset_index()



plt.figure(figsize = (15,35))



i = 0

for lake in lakes:

    plt.subplot(6,2, i+1)

    sns.barplot( x= 'Month' ,y = lake, data = month)

    plt.xticks(rotation = 45)

    plt.xlabel('Month')

    plt.ylabel(lake)

    plt.title(lake + ' vs Month', size = 10)

        

    plt.subplot(6,2, i+2)

    sns.lineplot(x = 'Month', y = lake, data = water_level, ci = False)

    plt.xticks(rotation = 45)

    plt.xlabel('Month')

    plt.ylabel('Water_level')

    plt.title(lake + ' vs Month', size = 10)

    plt.subplots_adjust(hspace = 0.5)

    i+=2



plt.show()
#Let's consider rain fall

rain_fall = pd.read_csv('../input/chennai_reservoir_rainfall.csv')

rain_fall.head(3)
#Do the same for water level

rain_fall.Date = pd.to_datetime(rain_fall.Date)

rain_fall['Year'] = rain_fall.Date.dt.year

rain_fall['Month'] = rain_fall.Date.dt.month

rain_fall['Total'] = rain_fall.POONDI + rain_fall.CHOLAVARAM + rain_fall.REDHILLS + rain_fall.CHEMBARAMBAKKAM



rain_fall.Month.replace([1,2,3,4,5,6,7,8,9,10,11,12],

             ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], inplace = True)



rain_fall.Month = pd.Categorical(water_level.Month,

                                   ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 

                                   ordered = True )



rain_fall.head(3)

#EDA - Water_level vs Rainfall 

year_rain_fall = rain_fall.groupby('Year').sum().reset_index()





plt.figure(figsize = (15,35))

i = 0

for lake in lakes:

    plt.subplot(6,2,i+1)

    sns.lineplot(x = 'Year', y = lake, data = year, ci = False)

    plt.xticks(rotation = 45)

    plt.xlabel('Year')

    plt.ylabel('Water_level')

    plt.title(lake + ' vs Water_level')



    

    plt.subplot(6,2,i+2)

    sns.lineplot(x = 'Year', y = lake, data = rain_fall, ci= False)

    plt.xticks(rotation = 45)

    plt.xlabel('Year')

    plt.ylabel('Rain_fall')

    plt.title(lake + ' vs Rain_fall')

    plt.subplots_adjust(hspace = 0.5)

    i+=2

plt.show()
#EDA - Water_level vs Rainfall 

month_rain_fall = rain_fall.groupby('Month').mean().reset_index()





plt.figure(figsize = (15,35))

i = 0

for lake in lakes:

    plt.subplot(6,2,i+1)

    sns.lineplot(x = 'Month', y = lake, data = month, ci = False)

    plt.xticks(rotation = 45)

    plt.xlabel('Month')

    plt.ylabel('Water_level')

    plt.title(lake + ' vs Water_level')



    

    plt.subplot(6,2,i+2)

    sns.lineplot(x = 'Month', y = lake, data = rain_fall, ci = False)

    plt.xticks(rotation = 45)

    plt.xlabel('Month')

    plt.ylabel('Rain_fall')

    plt.title(lake + ' vs Rain_fall')

    plt.subplots_adjust(hspace = 0.5)

    i+=2

plt.show()
yearly_water_level = water_level.groupby(['Year','Month']).mean().reset_index()

yearly_rain_fall = rain_fall.groupby(['Year','Month']).mean().reset_index()



years = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]



plt.figure(figsize = (10,100))

i= 0

for year in years:

    

    level = yearly_water_level[yearly_water_level.Year == year]

    rain =  yearly_rain_fall[yearly_rain_fall.Year == year]

    

    plt.subplot(16, 2, i+1)

    sns.barplot(x = 'Month', y = 'Total', data = level)

    plt.ylim(0,12000)

    plt.xticks(rotation = 45)

    plt.xlabel(year)

    plt.ylabel('Water Level')

    plt.title ( str(year) + ' water level')

    

    plt.subplot(16, 2, i+2)

    sns.lineplot(x = 'Month', y = 'Total', data = rain)

    plt.ylim(0,120)

    plt.xticks(rotation = 45)

    plt.xlabel(year)

    plt.ylabel('rain fall')

    plt.title ( str(year) + ' rain level')

    

    plt.subplots_adjust(hspace = 1)

    i+=2



plt.show()

    