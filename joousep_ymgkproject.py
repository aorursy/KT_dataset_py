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
import joblib

import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

# from sklearn.preprocessing import Imputer

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import requests
airQualityDF= pd.read_excel("/kaggle/input/istanbul2.xlsx")

airQualityDF.fillna(0,inplace=True)
airQualityDF.shape
listCities = ['Kandilli', 'Üsküdar', 'Sirinevler', 'Mecidiyekoy', 'Umraniye', 'Basaksehir', 'Esenyurt', 'Sultanbeyli','Kagithane', 'Sultangazi', 'Silivri', 'Sile']

hkiKandilli = list(airQualityDF.filter(regex = listCities[0]).columns)

hkiKandilli
def calculateAirQualityIndexSO2(so2):

    soi2=0

    if (so2>=0 and so2<=100):

     soi2= ((50-0)/(100-0))*(so2-0) + 0

    if (so2>=101 and so2<=250):

     soi2= ((100-51)/(250-101))*(so2-101) + 51

    if (so2>=251 and so2<=500):

     soi2= ((150-101)/(500-251))*(so2-251) + 101

    if (so2>=501 and so2<=850):

     soi2= ((200-151)/(850-501))*(so2-501) + 151

    if (so2>=851 and so2<=1100):

     soi2= ((300-201)/(1100-851))*(so2-851) + 201

    if (so2>=1101 and so2<= 1500):

     soi2= ((500-301)/(1500-1101))*(so2-1101) + 301

    

    return soi2
def calculateAirQualityIndexNo2(no2):

    noi2=0

    if (no2>=0 and no2<=100):

     noi2= ((50-0)/(100-0))*(no2-0) + 0

    if (no2>=101 and no2<=200):

     noi2= ((100-51)/(200-101))*(no2-101) + 51

    if (no2>=201 and no2<=500):

     noi2= ((150-101)/(500-201))*(no2-201) + 101

    if (no2>=501 and no2<=1000):

     noi2= ((200-151)/(1000-501))*(no2-501) + 151

    if (no2>=1001 and no2<=2000):

     noi2= ((300-201)/(2000-1001))*(no2-1001) + 201

    if (no2>=2001 and no2<= 3000):

     noi2= ((500-301)/(3000-2001))*(no2-2001) + 301

    return noi2
def calculateAirQualityIndexPM10(pm10):

    pm10i2=0

    if (pm10>=0 and pm10<=50):

     pm10i2= ((50-0)/(50-0))*(pm10-0) + 0

    if (pm10>=51 and pm10<=100):

     pm10i2= ((100-51)/(100-51))*(pm10-51) + 51

    if (pm10>=101 and pm10<=260):

     pm10i2= ((150-101)/(260-101))*(pm10-101) + 101

    if (pm10>=261 and pm10<=400):

     pm10i2= ((200-151)/(400-261))*(pm10-261) + 151

    if (pm10>=401 and pm10<=520):

     pm10i2= ((300-201)/(520-401))*(pm10-401) + 201

    if (pm10>=521 and pm10<= 620):

     pm10i2= ((500-301)/(620-521))*(pm10-521) + 301

    

    return pm10i2 
def calculateAirQualityIndexPM25(pm25):

    pm25i2=0

    if (pm25>=0 and pm25<=12):

     pm25i2= ((50-0)/(12-0))*(pm25-0) + 0

    if (pm25>=12.1 and pm25<=35.4):

     pm25i2= ((100-51)/(35.4-12.1))*(pm25-12.1) + 51

    if (pm25>=35.5 and pm25<=55.4):

     pm25i2= ((150-101)/(55.4-35.5))*(pm25-35.5) + 101

    if (pm25>=55.5 and pm25<=150.4):

     pm25i2= ((200-151)/(150.4-55.5))*(pm25-55.5) + 151

    if (pm25>=150.5 and pm25<=250.4):

     pm25i2= ((300-201)/(250.4-150.5))*(pm25-150.5) + 201

    if (pm25>=250.5 and pm25<= 350.4):

     pm25i2= ((400-301)/(350.4-250.5))*(pm25-250.5) + 301

    if (pm25>=350.5 and pm25<= 505.4):

     pm25i2= ((500-401)/(505.4-350.5))*(pm25-350.5) + 401

    return pm25i2 
def calculateAirQualityIndexCO(CO):

    coi2=0

    if (CO>=0 and CO<=5500):

     coi2= ((50-0)/(5500-0))*(CO-0) + 0

    if (CO>=5501 and CO<=10000):

     coi2= ((100-51)/(10000-5501))*(CO-5501) + 51

    if (CO>=10001 and CO<=16000):

     coi2= ((150-101)/(16000-10001))*(CO-10001) + 101

    if (CO>=16001 and CO<=24000):

     coi2= ((200-151)/(24000-16001))*(CO-16001) + 151

    if (CO>=24001 and CO<=32000):

     coi2= ((300-201)/(32000-24001))*(CO-24001) + 201

    if (CO>=32001 and CO<=40000):

     coi2= ((500-301)/(40000-32001))*(CO-32001) + 301

    return coi2 
def calculateAirQualityIndexO3(O3):

    o3i2=0

    if (O3>=0 and O3<=120):

     o3i2= ((50-0)/(120-0))*(O3-0) + 0

    if (O3>=121 and O3<=160):

     o3i2= ((100-51)/(160-121))*(O3-121) + 51

    if (O3>=161 and O3<=180):

     o3i2= ((150-101)/(180-161))*(O3-161) + 101

    if (O3>=181 and O3<=240):

     o3i2= ((200-151)/(240-181))*(O3-181) + 151

    if (O3>=241 and O3<=700):

     o3i2= ((300-201)/(700-241))*(O3-241) + 201

    if (O3>=701 and O3<=1700):

     o3i2= ((500-301)/(1700-701))*(O3-701) + 301

    return o3i2  
def calculateHKI():

    listPM10 = list(airQualityDF.filter(like='PM10').columns)

    listSO2 = list(airQualityDF.filter(like='SO2').columns)

    listNO2 = list(airQualityDF.filter(like='NO2').columns)

    listCO = list(airQualityDF.filter(like='CO').columns)

    listO3 = list(airQualityDF.filter(like='O3').columns)

    listPM25 = list(airQualityDF.filter(like='PM 2.5').columns)

    for pm10 in listPM10:

        airQualityDF["HKI"+pm10] = airQualityDF[pm10].apply(calculateAirQualityIndexPM10)

    for so2 in listSO2:

        airQualityDF["HKI"+so2] = airQualityDF[so2].apply(calculateAirQualityIndexSO2)

    for no2 in listNO2:

        airQualityDF["HKI"+no2] = airQualityDF[no2].apply(calculateAirQualityIndexNo2)

    for co in listCO:

        airQualityDF["HKI"+co] = airQualityDF[co].apply(calculateAirQualityIndexCO)

    for O3 in listO3:

        airQualityDF["HKI"+O3] = airQualityDF[O3].apply(calculateAirQualityIndexO3)

    for PM25 in listPM25:

        airQualityDF["HKI"+PM25] = airQualityDF[PM25].apply(calculateAirQualityIndexPM25)

calculateHKI()
def splitHKIValueAndValue(newList):

    listMaterialName=[]

    listHighValue = []

    for dongu in airQualityDF[newList].values:

        listMaterialName.append(newList[dongu.argmax()].split('-')[1])

        listHighValue.append(dongu.max())

    columnName = newList[0].split('HKI')[1].split('-')[0]

    columnType = columnName+'Type'

    airQualityDF['AQI-'+columnName] = listHighValue

    airQualityDF['AQI-'+columnType] = listMaterialName



hkiKandilli = list(airQualityDF.filter(regex = 'HKIKandilli-').columns)

hkiUskudar = list(airQualityDF.filter(regex = 'HKIÜsküdar-').columns)

hkiSirinevler = list(airQualityDF.filter(regex = 'HKISirinevler-').columns)

hkiMecidiyekoy = list(airQualityDF.filter(regex = 'HKIMecidiyekoy-').columns)

hkiUmraniye =list(airQualityDF.filter(regex = 'HKIUmraniye-').columns)

hkiBasaksehir = list(airQualityDF.filter(regex = 'HKIBasaksehir-').columns)

hkiEsenyurt = list(airQualityDF.filter(regex = 'HKIEsenyurt-').columns)

hkiSultanbeyli = list(airQualityDF.filter(regex = 'HKISultanbeyli-').columns)

hkiKagithane = list(airQualityDF.filter(regex = 'HKIKagithane-').columns)

hkiSultangazi = list(airQualityDF.filter(regex = 'HKISultangazi-').columns)

hkiSilivri = list(airQualityDF.filter(regex = 'HKISilivri-').columns)

hkiSile = list(airQualityDF.filter(regex = 'HKISile-').columns)



splitHKIValueAndValue(hkiKandilli)

splitHKIValueAndValue(hkiUskudar)

splitHKIValueAndValue(hkiSirinevler)

splitHKIValueAndValue(hkiMecidiyekoy)

splitHKIValueAndValue(hkiUmraniye)

splitHKIValueAndValue(hkiBasaksehir)

splitHKIValueAndValue(hkiEsenyurt)

splitHKIValueAndValue(hkiSultanbeyli)

splitHKIValueAndValue(hkiKagithane)

splitHKIValueAndValue(hkiSultangazi)

splitHKIValueAndValue(hkiSilivri)

splitHKIValueAndValue(hkiSile)
def calculateGoodOrBadAir(listValues):

    hkiString=""

    

    for value in listValues:

        listHKIString=[]

        for hkiValues in airQualityDF[value]:

            if(hkiValues>=0 and hkiValues<=50):

                hkiString = "İyi"

            if(hkiValues>=51 and hkiValues<=100):

                hkiString = "Orta"

            if(hkiValues>=101 and hkiValues<=150):

                hkiString = "Hassas"

            if(hkiValues>=151 and hkiValues<=200):

                hkiString = "Sağlıksız"

            if(hkiValues>=201 and hkiValues<=300):

                hkiString = "Kötü"

            if(hkiValues>=301 and hkiValues<=500):

                hkiString = "Tehlikeli"

            listHKIString.append(hkiString)

        airQualityDF['HKIStr-'+value.split('-')[1]] = listHKIString

aqiList = list(airQualityDF.filter(regex = 'AQI-').columns)[::2]

calculateGoodOrBadAir(aqiList)
kandilli = airQualityDF.loc[:,[hkiKandilli[0], hkiKandilli[1], hkiKandilli[2], hkiKandilli[3], hkiKandilli[4], hkiKandilli[5], 'AQI-Kandilli','AQI-KandilliType']]

plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(kandilli['AQI-Kandilli'])

plt.show()
airQualityDF.isnull().any()
f,ax=plt.subplots(figsize=(15,8))

plt.plot(airQualityDF['Tarih'], airQualityDF['Sirinevler-PM10'],color='r', label='PM10',alpha=0.8)

plt.plot(airQualityDF['Tarih'],airQualityDF['Sirinevler-SO2'],color='b', label='SO2',alpha=0.8)

plt.legend(loc='upper left')

plt.show()
winterMonths = []

winterValues = []

summerMonths = []

summerValues = []

for days in airQualityDF['Tarih']:

    if(days.month == 12 or days.month == 1 or days.month == 2):

        winterValues.append(airQualityDF[airQualityDF['Tarih']==days]['Sirinevler-PM10'].values[0])

        winterMonths.append(days)

    if(days.month == 6 or days.month == 7 or days.month == 8):

        summerValues.append(airQualityDF[airQualityDF['Tarih']==days]['Sirinevler-PM10'].values[0])

        summerMonths.append(days)

f,ax=plt.subplots(figsize=(15,8))

plt.plot(winterMonths, winterValues,color='r', label='winter',alpha=0.8)

plt.plot(summerMonths,summerValues, color='b', label='summer',alpha=0.8)

plt.title('Yaz ve Kış Ayları')

plt.legend(loc='upper left')

plt.show()
dailyPM10 = []

dailyPM10Values = []

for year in airQualityDF['Tarih']:

    if(year.year == 2020 and year.month == 4 and year.day == 1):

        dailyPM10Values.append(airQualityDF[airQualityDF['Tarih']==year]['Umraniye-PM10'].values[0])

        dailyPM10.append(year)
f,ax=plt.subplots(figsize=(15,8))

plt.plot(dailyPM10, dailyPM10Values ,color='r', label='PM10',alpha=0.8)

plt.title('Bir Günlük Ümraniye İlçesi PM10 Değişimi')

plt.legend(loc='upper left')

plt.show()
dailyPM10 = []

dailyPM10Values = []

for year in airQualityDF['Tarih']:

    if(year.year == 2020 and year.month == 4 and year.day == 1):

        dailyPM10Values.append(airQualityDF[airQualityDF['Tarih']==year]['Basaksehir-PM10'].values[0])

        dailyPM10.append(year)

dailySO2 = []

dailySO2Values = []

for year in airQualityDF['Tarih']:

    if(year.year == 2020 and year.month == 4 and year.day == 1):

        dailySO2Values.append(airQualityDF[airQualityDF['Tarih']==year]['Basaksehir-SO2'].values[0])

        dailySO2.append(year)

dailyO3 = []

dailyO3Values = []

for year in airQualityDF['Tarih']:

    if(year.year == 2020 and year.month == 4 and year.day == 1):

        dailyO3Values.append(airQualityDF[airQualityDF['Tarih']==year]['Basaksehir-O3'].values[0])

        dailyO3.append(year)

dailyNO2 = []

dailyNO2Values = []

for year in airQualityDF['Tarih']:

    if(year.year == 2020 and year.month == 4 and year.day == 1):

        dailyNO2Values.append(airQualityDF[airQualityDF['Tarih']==year]['Basaksehir-NO2'].values[0])

        dailyNO2.append(year)



f,ax=plt.subplots(figsize=(15,8))

plt.plot(dailySO2, dailySO2Values ,color='r', label='SO2',alpha=0.8)

plt.title('Bir Günlük Başakşehir İlçesi SO2 Değişimi')

plt.legend(loc='upper left')

plt.show()

f,ax=plt.subplots(figsize=(15,8))

plt.plot(dailyPM10, dailyPM10Values ,color='r', label='PM10',alpha=0.8)

plt.title('Bir Günlük Başakşehir İlçesi PM10 Değişimi')

plt.legend(loc='upper left')

plt.show()
f,ax=plt.subplots(figsize=(15,8))

plt.plot(dailyNO2, dailyNO2Values ,color='r', label='NO2',alpha=0.8)

plt.title('Bir Günlük Başakşehir İlçesi NO2 Değişimi')

plt.legend(loc='upper left')

plt.show()
f,ax=plt.subplots(figsize=(15,8))

plt.plot(dailyO3, dailyO3Values ,color='r', label='O3',alpha=0.8)

plt.title('Bir Günlük Başakşehir İlçesi O3 Değişimi')

plt.legend(loc='upper left')

plt.show()