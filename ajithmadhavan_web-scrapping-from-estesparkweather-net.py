import os

print(os.listdir("../input"))



import bs4

from bs4 import BeautifulSoup

import csv

import requests

import time

import pandas as pd

import urllib

import re

import pickle

import urllib.request



import arrow

import datetime

import pandas as pd
def get_date(param , day) :

    dt = param +day 

    a = arrow.get(dt, 'YYYYMD').date()

    return str(a)



period_temp=pd.date_range(start='1/1/2009',end='11/1/2018',freq='M')

periods=[]

for period in period_temp:

    yearMonth=str(period)[:4]+str(period)[5:7]

    periods.append(yearMonth)

#print(periods)



monthwise_data=[]

def createDataFrameContents(final,param,month_text):

    for record in final:

        daywiseData=[]

        for item in record:

            if item.startswith(month_text):

                day=item.split(' ')[1]

                if len(day)==1:

                    day='0'+day

                #print(get_date(param,day))

                daywiseData.append(get_date(param,day))

            if item.startswith('Average temperature'):

                avgTemp=item.split(' ')[2]

                #print(float(avgTemp))

                daywiseData.append(avgTemp)

            if item.startswith('Average humidity'):

                avgHum=item.split(' ')[2]

                #print(int(avgHum))

                daywiseData.append(avgHum)

            if item.startswith('Average dewpoint'):

                avgDewPt=item.split(' ')[2]

                #print(float(avgDewPt))

                daywiseData.append(avgDewPt)

            if item.startswith('Average barometer'):

                avgBaro=item.split(' ')[2]

                #print(float(avgBaro))

                daywiseData.append(avgBaro)

            if item.startswith('Average windspeed'):

                avgWindSpeed=item.split(' ')[2]

                #print(float(avgWindSpeed))

                daywiseData.append(avgWindSpeed)

            if item.startswith('Average gustspeed'):

                avgGustSpeed=item.split(' ')[2]

                #print(float(avgGustSpeed))

                daywiseData.append(avgGustSpeed)

            if item.startswith('Average direction'):

                avgDir=item.split(' ')[2].split('°')[0]

                #print(int(avgDir))

                daywiseData.append(avgDir)

            if item.startswith('Rainfall for month'):

                rainFM=item.split(' ')[3]

                #print(float(rainFM))

                daywiseData.append(float(rainFM))

            if item.startswith('Rainfall for year'):

                rainFY=item.split(' ')[3]

                #print(float(rainFY))

                daywiseData.append(rainFY)

            if item.startswith('Maximum rain per minute'):

                maxRPM=item.split(' ')[4]

                #print(float(maxRPM))

                daywiseData.append(maxRPM)

            if item.startswith('Maximum temperature'):

                maxTemp=item.split(' ')[2].split('°')[0]

                #print(float(maxTemp))

                daywiseData.append(maxTemp)

            if item.startswith('Minimum temperature'):

                minTemp=item.split(' ')[2].split('°')[0]

                #print(float(minTemp))

                daywiseData.append(minTemp)

            if item.startswith('Maximum humidity'):

                maxHum=item.split(' ')[2].split('%')[0]

                #print(int(maxHum))

                daywiseData.append(maxHum)

            if item.startswith('Minimum humidity'):

                minHum=item.split(' ')[2].split('%')[0]

                if len(minHum)==0:

                    minHum=item.split(' ')[3].split('%')[0]

                #print(int(minHum))

                daywiseData.append(minHum)

            if item.startswith('Maximum pressure'):

                maxPres=item.split(' ')[2]

                #print(float(maxPres))

                daywiseData.append(maxPres)

            if item.startswith('Minimum pressure'):

                minPres=item.split(' ')[2]

                #print(float(minPres))

                daywiseData.append(minPres)

            if item.startswith('Maximum windspeed'):

                maxWindSpeed=item.split(' ')[2]

                #print(float(maxWindSpeed))

                daywiseData.append(maxWindSpeed)

            if item.startswith('Maximum gust speed'):

                maxGS=item.split(' ')[3]

                #print(int(maxGS))

                daywiseData.append(maxGS)

            if item.startswith('Maximum heat index'):

                maxHI=item.split(' ')[3]

                #print(float(maxHI))

                daywiseData.append(maxHI)

        monthwise_data.append(daywiseData)

    return monthwise_data



month_text =['Jan', 'Feb' ,'Mar', 'Apr' ,'May' ,'Jun', 'Jul','Aug' ,'Sep','Oct','Nov','Dec']



for param in periods[:]:

    

    #param='200901'

    page=urllib.request.urlopen('http://www.estesparkweather.net/archive_reports.php?date='+ param)

    soup = BeautifulSoup(page, 'lxml')

    table_container = soup.find_all('table')

    data=[]

    for row in table_container:

        data.append(row.text.splitlines())



    for x in data:

        while '' in x:

            x.remove('')

    index=param[4:6]

    final_index=int(index)-1

    final=[]

    for x in data:

        if x[0].startswith(month_text[final_index]):

            final.append(x)

    createDataFrameContents(final,param,month_text[final_index])

    

#print(type(table_container))

df_index=[]

avg_temp,avg_Hum,avg_DewPt,avg_Baro,avg_WS,avg_GS,avg_Dir=[],[],[],[],[],[],[]

rfm,rfy,mrpm,maxTemp,minTemp,maxHum,minHum,maxPres,minPres,maxWS,maxGP,maxHI=[],[],[],[],[],[],[],[],[],[],[],[]



if len(monthwise_data)>0:

    for rows in monthwise_data:

        df_index.append(rows[0])

        avg_temp.append(rows[1])

        avg_Hum.append(rows[2])

        avg_DewPt.append(rows[3])

        avg_Baro.append(rows[4])

        avg_WS.append(rows[5])

        avg_GS.append(rows[6])

        avg_Dir.append(rows[7])

        rfm.append(rows[8])

        rfy.append(rows[9])

        mrpm.append(rows[10])

        maxTemp.append(rows[11])

        minTemp.append(rows[12])

        maxHum.append(rows[13])

        minHum.append(rows[14])

        maxPres.append(rows[15])

        minPres.append(rows[16])

        maxWS.append(rows[17])

        maxGP.append(rows[18])

        maxHI.append(rows[19])

df=pd.DataFrame({'Average temperature':avg_temp,'Average humidity':avg_Hum,'Average dewpoint':avg_DewPt,

                      'Average barometer':avg_Baro,'Average windspeed':avg_WS,'Average gustspeed':avg_GS,

                      'Average direction':avg_Dir,'Rainfall for month':rfm,'Rainfall for year':rfy,

                      'Maximum rain per minute':mrpm,'Maximum temperature':maxTemp,'Minimum temperature':minTemp,

                     'Maximum humidity':maxHum,'Minimum humidity':minHum,'Maximum pressure':maxPres,

                     'Minimum pressure':minPres,'Maximum windspeed':maxWS,'Maximum gust speed':maxGP,

                     'Maximum heat index':maxHI},index=df_index)

print(df.head(100))

df['Average temperature']=df['Average temperature'].str.strip('°F%')

df['Average humidity']=df['Average humidity'].str.strip('°F%')

df['Average dewpoint']=df['Average dewpoint'].str.strip('°F%')

df['Maximum heat index']=df['Maximum heat index'].str.strip('°F%')

df['dfindex'] = pd.to_datetime(df['dfindex'])

df.set_index('dfindex', inplace=True)

end1=time.time()

print(end1-start1)

print('Great job! Web Scrapping is successfully completed! ')

#pd.to_datetime(df.index)

df= df.apply(pd.to_numeric)

df.to_csv('updatedFileCS.csv')

#df.index.to_datetime()