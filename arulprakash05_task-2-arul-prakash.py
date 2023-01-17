# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as math

import random as rand

import matplotlib.pyplot as plt

import fbprophet as fb

import datetime as dt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing from CSV

dfGen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

dfSensor1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')



#Converting DATETIME into Date and Time



#DfGen1

#Date

dfGen1['DATE_TIME'] = pd.to_datetime(dfGen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')   

dfGen1['DATE'] = pd.to_datetime(dfGen1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date  

dfGen1['DATE'] = pd.to_datetime(dfGen1['DATE'] )

#Time

dfGen1['TIME'] = pd.to_datetime(dfGen1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time   

dfGen1['TIME'] = pd.to_datetime(dfGen1['TIME'], format = '%H:%M:%S')

dfGen1.info()



#DfSensor1

#Date

dfSensor1['DATE_TIME'] = pd.to_datetime(dfSensor1['DATE_TIME'],format = '%Y-%m-%d %H:%M')  

dfSensor1['DATE'] = pd.to_datetime(dfSensor1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date   

dfSensor1['DATE'] = pd.to_datetime(dfSensor1['DATE'] )

#Time

dfSensor1['TIME'] = pd.to_datetime(dfSensor1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time   

dfSensor1['TIME'] = pd.to_datetime(dfSensor1['TIME'], format = '%H:%M:%S')

dfSensor1.info()
dfMerged = pd.merge(dfGen1,dfSensor1, on =['DATE_TIME', 'DATE', 'TIME'], how='left')

dfMerged.info()
DC_POW = dfMerged.groupby('DATE')['DC_POWER'].sum()

uniqueDate = dfMerged['DATE'].unique()

plt.figure(figsize = (21, 12))

for i in range(0, dfMerged['DATE'].nunique()):

    plt.barh(uniqueDate[i], DC_POW[i], color = 'green')

plt.xlabel('DC POWER')

plt.ylabel('DATE')

plt.title('DC POWER PER DAY')

plt.show()
DC_POW = dfMerged.groupby('DATE_TIME')['DC_POWER'].sum()

DateTime = dfMerged['DATE_TIME'].unique()

plt.figure(figsize = (21,12))

plt.plot(DateTime, DC_POW.rolling(window = 9).mean())

plt.xlabel('DATE AND TIME')

plt.ylabel('DC POWER')

plt.title('DC POWER CHART')

plt.grid()

plt.show()
DC_POW = dfMerged.groupby('TIME')['DC_POWER'].mean()

p = dfMerged['DATE_TIME'].dt.time

uniqueTIME = p.unique()

for o in range(0,24*4):

    uniqueTIME[o] = str(uniqueTIME[o])

plt.figure(figsize  = (21, 12))

plt.bar(uniqueTIME, DC_POW, color = 'Red')

TimeList = list()

for i in range(0,24*4, 4):     # 24 hours, 4 readings per hour

    TimeList.append(uniqueTIME[i])

#print(TimeList)

plt.xticks(TimeList, rotation = 90)

plt.xlabel('TIME')

plt.ylabel('DC POWER')

plt.title('DC POWER BY A GENERATOR ON AN AVERAGE DAY')

plt.show()
AC_POW = dfMerged.groupby('DATE')['AC_POWER'].sum()

uniqueDate = dfMerged['DATE'].unique()

plt.figure(figsize = (21, 12))

for i in range(0, dfMerged['DATE'].nunique()):

    plt.barh(uniqueDate[i], AC_POW[i], color = 'maroon')

plt.xlabel('AC POWER')

plt.ylabel('DATE')

plt.title('AC POWER PER DAY')

plt.show()
AC_POW = dfMerged.groupby('DATE_TIME')['AC_POWER'].sum()

DateTime = dfMerged['DATE_TIME'].unique()

plt.figure(figsize = (21,12))

plt.plot(DateTime, AC_POW.rolling(window = 9).mean(), color = 'k')

plt.xlabel('DATE and TIME')

plt.ylabel('AC POWER')

plt.title('AC POWER CHART')

plt.grid()

plt.show()
AC_POW = dfMerged.groupby('TIME')['AC_POWER'].mean()

p = dfMerged['DATE_TIME'].dt.time

uniqueTIME = p.unique()

for o in range(0,24*4):

    uniqueTIME[o] = str(uniqueTIME[o])

plt.figure(figsize  = (21, 12))

plt.bar(uniqueTIME, AC_POW, color = 'purple')

TimeList = list()

for i in range(0,24*4, 4):     # 24 hours, 4 readings per hour

    TimeList.append(uniqueTIME[i])

#print(TimeList)

plt.xticks(TimeList, rotation = 90)

plt.xlabel('TIME')

plt.ylabel('AC POWER')

plt.title('AC POWER BY A GENERATOR ON AN AVERAGE DAY')

plt.show()
IrrP = dfMerged.groupby('DATE')['IRRADIATION'].sum()

unD = dfMerged['DATE'].unique()

plt.figure(figsize = (21, 12))

plt.barh(unD, IrrP, color = 'Blue')

plt.xlabel('IRRADIATION')

plt.ylabel('DATE')

plt.title('IRRADIATION PER DAY')

plt.show()


IrrP = dfMerged.groupby('DATE_TIME')['IRRADIATION'].mean()

DateTime = dfMerged['DATE_TIME'].unique()

plt.figure(figsize = (21,12))

plt.plot(DateTime, IrrP.rolling(window = 9).mean(), color = 'green')

plt.xlabel('DATE and TIME')

plt.ylabel('IRRADIATION')

plt.title('IRRADIATION CHART')

plt.grid()

plt.show()
IrrP = dfMerged.groupby('TIME')['IRRADIATION'].mean()

p = dfMerged['DATE_TIME'].dt.time

uniqueTIME = p.unique()



for o in range(0,24*4):

    uniqueTIME[o] = str(uniqueTIME[o])



TimeList = list()

for i in range(0,24*4, 4):     # 24 hours, 4 readings per hour

    TimeList.append(uniqueTIME[i])



plt.figure(figsize  = (21, 12))

plt.bar(uniqueTIME, IrrP, color = 'black')



plt.xticks(TimeList, rotation = 90)

plt.xlabel('TIME')

plt.ylabel('IRRADIATION')

plt.title('IRRADIATION ON AN AVERAGE DAY')

plt.show()
Amb = dfMerged.groupby('DATE_TIME')['AMBIENT_TEMPERATURE'].mean()

DateTime = dfMerged['DATE_TIME'].unique()

plt.figure(figsize = (21,12))

plt.plot(DateTime, Amb.rolling(window = 9).mean(), color = 'red')

plt.xlabel('DATE and TIME')

plt.ylabel('AMBIENT TEMPERATURE')

plt.title('AMBIENT TEMPERATURE CHART')

plt.grid()

plt.show()
Amb = dfMerged.groupby('TIME')['AMBIENT_TEMPERATURE'].mean()

p = dfMerged['DATE_TIME'].dt.time

uniqueTIME = p.unique()



for o in range(0,24*4):

    uniqueTIME[o] = str(uniqueTIME[o])



TimeList = list()

for i in range(0,24*4, 4):     # 24 hours, 4 readings per hour

    TimeList.append(uniqueTIME[i])



plt.figure(figsize  = (21, 12))

plt.bar(uniqueTIME, Amb, color = 'blue')



plt.xticks(TimeList, rotation = 90)

plt.xlabel('TIME')

plt.ylabel('AMBIENT TEMPERATURE')

plt.title('AMBIENT TEMPERATURE ON AN AVERAGE DAY')

plt.show()
Modu = dfMerged.groupby('DATE_TIME')['MODULE_TEMPERATURE'].mean()

DateTime = dfMerged['DATE_TIME'].unique()

plt.figure(figsize = (21,12))

plt.plot(DateTime, Modu.rolling(window = 9).mean(), color = 'purple')

plt.xlabel('DATE and TIME')

plt.ylabel('MODULE TEMPERATURE')

plt.title('MODULE TEMPERATURE CHART')

plt.grid()

plt.show()
Modu = dfMerged.groupby('TIME')['MODULE_TEMPERATURE'].mean()

p = dfMerged['DATE_TIME'].dt.time

uniqueTIME = p.unique()



for o in range(0,24*4):

    uniqueTIME[o] = str(uniqueTIME[o])



TimeList = list()

for i in range(0,24*4, 4):     # 24 hours, 4 readings per hour

    TimeList.append(uniqueTIME[i])



plt.figure(figsize  = (21, 12))

plt.bar(uniqueTIME, Modu, color = 'orange')



plt.xticks(TimeList, rotation = 90)

plt.xlabel('TIME')

plt.ylabel('MODULE TEMPERATURE')

plt.title('MODULE TEMPERATURE ON AN AVERAGE DAY')

plt.show()
TotalY = dfMerged.groupby('DATE_TIME')['TOTAL_YIELD'].sum()

DateTime = dfMerged['DATE_TIME'].unique()

plt.figure(figsize = (21, 12))

plt.plot(DateTime, TotalY.rolling(window = 14).mean(), color = 'red', linewidth = 2.5)

plt.xlabel('DATETIME')

plt.ylabel('TOTAL YIELD')

plt.title('CHART OF TOTAL YIELD')

plt.grid()

plt.show()
TotalY = dfMerged.groupby('DATE')['TOTAL_YIELD'].max()

uniqueDate = dfMerged['DATE'].unique()

plt.figure(figsize = (21, 12))

plt.bar(uniqueDate, TotalY, color = "violet")

plt.xlabel('TOTAL YIELD')

plt.ylabel('DATE')

plt.title('TOTAL YIELD AT THE END OF THE DAY')

plt.grid()

plt.show()
DC_POW = dfMerged.iloc[:,[3]].values

AC_POW = dfMerged.iloc[:,[4]].values

plt.figure(figsize = (21, 12))

plt.scatter(DC_POW, AC_POW, color = 'Blue')

plt.xlabel('DC POWER')

plt.ylabel('AC POWER')

plt.title('DC POWER VS AC POWER')

plt.grid()

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(DC_POW, AC_POW, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

plt.figure(figsize = (21, 12))

plt.plot(x_test, y_test, color = 'Blue', linewidth = 1, label = 'Actual Outcome')

plt.plot(x_test, model.predict(x_test), color = 'red', linewidth = 1, label = 'Predicted Outcome')

plt.legend(loc = 'lower right')

plt.xlabel('DC POWER')

plt.ylabel('AC POWER')

plt.title('MODEL TEST')

plt.grid()

plt.show()
data1 = pd.DataFrame({'DC_POWER':np.random.randint(0, 13000, 100)})

data1 = data1.iloc[:, [0]].values

prediction = model.predict(data1)

plt.figure(figsize = (21, 12))

plt.plot(x_test, y_test, color = 'Blue', label = 'Expected Line')

plt.plot(data1, prediction, color = 'red', label = 'Model Predictions')

plt.legend(loc = 'lower right')

plt.xlabel('DC POWER')

plt.ylabel('AC POWER')

plt.title('MODEL RUN')

plt.grid()

plt.show()
Amb = dfMerged.groupby('DATE_TIME')['AMBIENT_TEMPERATURE'].mean()

Modu = dfMerged.groupby('DATE_TIME')['MODULE_TEMPERATURE'].mean()



plt.figure(figsize = (21, 12))

plt.scatter(Amb, Modu, color = 'purple')

plt.xlabel('AMBIENT TEMPERATURE')

plt.ylabel('MODULE TEMPERATURE')

plt.title('AMBIENT TEMPERATURE VS MODULE TEMPERATURE')

plt.show()
Amb = dfMerged.groupby('DATE_TIME')['AMBIENT_TEMPERATURE'].mean()

Irr = dfMerged.groupby('DATE_TIME')['IRRADIATION'].mean()



plt.figure(figsize = (21, 12))

plt.scatter(Amb, Irr, color = 'red')

plt.xlabel('AMBIENT TEMPERATURE')

plt.ylabel('IRRADIATION')

plt.title('IRRADIATION VS AMBIENT TEMPERATURE')

plt.show()
Modu = dfMerged.groupby('DATE_TIME')['MODULE_TEMPERATURE'].mean()

Irr = dfMerged.groupby('DATE_TIME')['IRRADIATION'].mean()



plt.figure(figsize = (21, 12))

plt.scatter(Modu, Irr, color = 'orange')

plt.xlabel('MODULE TEMPERATURE')

plt.ylabel('IRRADIATION')

plt.title('MODULE TEMPERATURE VS IRRADIATION')

plt.show()
DC_POW = dfMerged['DC_POWER']

Irr = dfMerged['IRRADIATION']

plt.figure(figsize = (21, 12))

plt.scatter(DC_POW, Irr, color = 'purple', alpha = 0.7)

plt.xlabel('DC POWER')

plt.ylabel('IRRADIATION')

plt.title('DC POWER VS IRRADIATION')

plt.show()
AC_POW = dfMerged['AC_POWER']

Irr = dfMerged['IRRADIATION']

plt.figure(figsize = (21, 12))

plt.scatter(AC_POW, Irr, color = 'green', alpha = 0.7)

plt.xlabel('AC POWER')

plt.ylabel('IRRADIATION')

plt.title('AC POWER VS IRRADIATION')

plt.show()
DC_POW = dfMerged.groupby('DATE_TIME')['DC_POWER'].sum()

AC_POW = dfMerged.groupby('DATE_TIME')['AC_POWER'].sum()

DateTime = dfMerged['DATE_TIME'].unique()



plt.figure(figsize = (21,12))

plt.plot(DateTime, DC_POW.rolling(window = 9).mean(), label = 'DC POWER')

plt.plot(DateTime, AC_POW.rolling(window = 9).mean(), label = 'AC POWER')

plt.legend()

plt.xlabel('DATE AND TIME')

plt.ylabel('DC/AC POWER')

plt.title('DC/AC OVER TIME')

plt.grid()

plt.show()
Amb = dfMerged.groupby('DATE_TIME')['AMBIENT_TEMPERATURE'].mean()

Modu = dfMerged.groupby('DATE_TIME')['MODULE_TEMPERATURE'].mean()

DateTime = dfMerged['DATE_TIME'].unique()



plt.figure(figsize = (21, 12))

plt.plot(DateTime, Amb.rolling(window = 9).mean(), color = 'red', label = 'AMBIENT TEMPERATURE')

plt.plot(DateTime, Modu.rolling(window = 9).mean(), color = 'black', label = 'MODULE TEMPERATURE')

plt.xlabel('DATE AND TIME')

plt.ylabel('TEMPERATURE')

plt.legend()

plt.title('AMBIENT TEMPERATURE AND MODULE TEMPERATURE OVER TIME')

plt.grid()

plt.show()