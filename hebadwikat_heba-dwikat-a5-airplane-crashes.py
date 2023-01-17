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
#Loading libraries 



from datetime import date, timedelta, datetime

import seaborn as sns #apparently, this create more attractive graphs

import pandas_datareader.data as web

%matplotlib inline

import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime

#first i would like to check the columns that this file have and check which column i want to keep

air_crashes = pd.read_csv('/kaggle/input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv')

#Checking columns and rows number

print ('This air crashes dataset has {0} crashes and {1} features'.format(air_crashes.shape[0],air_crashes.shape[1]))

air_crashes.head()
#I would like to check the entries from each column to check the missing values



air_crashes.describe()

air_crashes.info()
air_crashes.isnull().sum()

#easier than doing the math in my head
#dropping the column im not interested in cn/ln : plane serial number,  flight # , time , route , registration

air_crashes.drop(columns=['Route', 'Time' , 'Flight #','cn/In','Registration', 'Summary'], axis=1, inplace=True)

air_crashes.head()

#air_crashes['Date'] = pd.DatetimeIndex(air_crashes['Date']).year
#checking the missing values again



air_crashes.isnull().sum()
#the amount of missing values isnt alot,I would like to get rid of these values,Im not sure if im doing the right thing



airDf = air_crashes.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

airDf.info()

#ok I think this is somehow cleaner , I didnt lose as much datapoints

airDf.isnull().sum()
airDf.head()
#checking the mean of fatalities and Aboard the last five years of the data

YearTemp=airDf

YearTemp['Year'] = pd.DatetimeIndex(YearTemp['Date']).year

YearTemp['Date'] = pd.to_datetime(YearTemp['Date'])

YearTemp = YearTemp.groupby(['Year']).mean()

YearTemp.tail()

Temp = airDf

Temp['Year'] = pd.DatetimeIndex(Temp['Date']).year #reducing the date to the year



Temp = Temp.groupby('Year')['Year'].count() #Temp is going to be temporary data frame 

Temp = Temp.rename(columns={"Year": "Count"})

                       

Temp.plot()

plt.xlabel('Year', fontsize=10)

plt.ylabel('Count of crashes', fontsize=10)

plt.title('airplane crashes by year', loc='Center', fontsize=14)

plt.show()
Temp = airDf

Temp['Year'] = pd.DatetimeIndex(Temp['Date']).year #reducing the date to the year



Temp = Temp.groupby('Year')['Fatalities', 'Aboard'].sum() #Temp is going to be temporary data frame 



Temp.plot(y = ['Fatalities', 'Aboard'])

plt.xlabel('Year', fontsize=10)

plt.ylabel('Count', fontsize=10)

plt.title('count Fatalities and Abroad / year ', loc='Center', fontsize=14)

plt.show()

Temp = airDf

Temp['Year'] = pd.DatetimeIndex(Temp['Date']).year

Temp['YearGroup'] = (Temp['Year']//5) * 5 #I would like t provide the survival rate every five years 



Temp = Temp.groupby('YearGroup')['Fatalities', 'Aboard'].sum() 

Temp['Survival'] = (Temp['Aboard'] - Temp['Fatalities']) / Temp['Aboard'] * 100



Temp.plot(y='Survival', figsize=(12,5))

plt.title('Survival Percantage throughout the years ', loc='Center', fontsize=14)

plt.xlabel('Year', fontsize=10)

plt.ylabel('Survival %', fontsize=10)



#Regarding this graph , I am not sure why the survival rate was very high at the beginning , mostly because airplanes passangers # early on was very small ,

#this shows me there might have been some noisy data between 1908 - 1913 that i could have tackled better
CrashesByPlaneType = airDf.groupby('Type')['Fatalities', 'Aboard'].sum()

CrashesByPlaneType = CrashesByPlaneType.sort_values(by='Fatalities', ascending=False)



CrashesByPlaneType[:5].plot.pie(y='Fatalities', figsize=(8,8) ,colors = ['firebrick', 'lightskyblue' ,'y' , 'darkseagreen' , 'darkolivegreen']) # I would like to check for the top five planes that had fatalities

plt.title('Total number of Fatalities by Type of flight', loc='Center', fontsize=14)

SurvivalByType = airDf.groupby('Type')['Fatalities', 'Aboard'].sum()

SurvivalByType = SurvivalByType[SurvivalByType.Fatalities != 0.0] #This will prevent the flights that had zero casualities from showing in this graph as we are interested in checking planes that had both fatalities and survivials but was overall safer than other planes

SurvivalByType['Survival'] = (SurvivalByType['Aboard'] - SurvivalByType['Fatalities']) / SurvivalByType['Aboard']

SurvivalByType = SurvivalByType.sort_values(by='Survival', ascending=False)

SurvivalByType.head()

SurvivalByType[:5].plot.pie(y='Survival', figsize=(8,8),colors = ['darkolivegreen','lightskyblue' ,'y' , 'darkseagreen' ,'yellow'])
Op_total = airDf.groupby('Operator')[['Operator']].count() #Since I have been checking fatalities mostly ,I would like to check the count of crashes by Operator

Op_total = Op_total.rename(columns={"Operator": "Count"}) #rename so i can use the "sum of Operators" as the count of crashes

Op_total = Op_total.sort_values(by='Count', ascending=False).head(10)



plt.figure(figsize=(12,6))

sns.barplot(y=Op_total.index, x="Count", data=Op_total,palette="Blues_d",orient='h')

plt.xlabel('Count', fontsize=10)

plt.ylabel('Operator', fontsize=10)

plt.title('Total Count by Opeartor', loc='Center', fontsize=14)

plt.show()

#Interesting , Aeroflot seemed to be the most fatal throughout the past 100 years, Military US force make sense to be the second as alot of these datapoints trace back to world war 2