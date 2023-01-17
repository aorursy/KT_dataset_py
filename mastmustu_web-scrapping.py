# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

max_p_range_day = re.findall(r'max_p_range_day\s*=\s* \'\d*[-,]?\d*[-,]?\d*\'',"max_p_range_day = '2018-03-23'")[0].replace(' ', '')

print(max_p_range_day)
period_date =pd.date_range(start='1/1/2009', end='8/1/2020', freq='M')

period =[]

for x in period_date :

    ym = str(x)[:4] + str(x)[5:7]

    period.append(ym)

    

print(period)



for x in period :

    index = x[4:6]

    #print(int(index)-1 )
import arrow

import datetime



def get_date(param , day) :

    dt = param +day 

    a = arrow.get(dt, 'YYYYMD').date()

    return str(a)





print(str(datetime.datetime.strptime("20131125", '%Y%m%d'))[:10])
month_data_list =[]



def get_components (records , param , month_text) :

    

    

    for record in records:

        days_record  =[]

        for x in record :

            

            if x.startswith(month_text) :

                day = x.split(' ')[1]

                if len(day) ==1 :

                    day ='0' + day

                #print(get_date(param ,day))

                index_column =get_date(param ,day)

                days_record.append(index_column)

            if x.startswith('Average temperature') :

                avg_temp = x.split('Average temperature')[1]

                #print(avg_temp[:-2])

                avg_temp = avg_temp [:-2]

                days_record.append(avg_temp)

            if x.startswith('Average humidity') :

                avg_humidity = x.split('Average humidity')[1]

                #print(avg_humidity[:-1])

                avg_humidity = avg_humidity [:-1]

                days_record.append(avg_humidity)

            if x.startswith('Average dewpoint') :

                avg_dewpoint = x.split('Average dewpoint')[1]

                #print(avg_dewpoint[:-2])

                avg_dewpoint = avg_dewpoint [:-2]

                days_record.append(avg_dewpoint)

            if x.startswith('Average barometer') :

                avg_baro = x.split('Average barometer')[1]

                #print(avg_baro[:-3])

                avg_baro = avg_baro [:-3]

                days_record.append(avg_baro)

            if x.startswith('Average windspeed') :

                avg_wind_speed = x.split('Average windspeed')[1]

                #print(avg_wind_speed[:-4])

                avg_wind_speed = avg_wind_speed [:-4]

                days_record.append(avg_wind_speed)

            if x.startswith('Average gustspeed') :

                avg_gust_speed = x.split('Average gustspeed')[1]

                #print(avg_gust_speed[:-4])

                avg_gust_speed = avg_gust_speed [:-4]

                days_record.append(avg_gust_speed)

            if x.startswith('Average direction') :

                avg_dir = x.split('Average direction')[1]

                #print(avg_dir[:-7])

                avg_dir = avg_dir [:-7]

                days_record.append(avg_dir)

            if x.startswith('Rainfall for month') :

                rainfall_per_month = x.split('Rainfall for month')[1]

                #print(rainfall_per_month[:-3])

                rainfall_per_month = rainfall_per_month [:-3]

                days_record.append(rainfall_per_month)

            if x.startswith('Rainfall for year') :

                rainfall_per_year = x.split('Rainfall for year')[1]

                #print(rainfall_per_year[:-3])

                rainfall_per_year = rainfall_per_year [:-3]

                days_record.append(rainfall_per_year)

            if x.startswith('Maximum rain per minute') :

                rainfall_per_min = x.split(' ')[4]

                #print(rainfall_per_min)

                rainfall_per_min = rainfall_per_min [:-3]

                days_record.append(rainfall_per_min)

            if x.startswith('Maximum temperature') :

                max_temp = x.split(' ')[2]

                #print(max_temp[:-2])

                max_temp = max_temp[:-2]

                days_record.append(max_temp)

            if x.startswith('Minimum temperature') :

                min_temp = x.split(' ')[2]

                #print(min_temp[:-2])

                min_temp = min_temp[:-2]

                days_record.append(min_temp)

            if x.startswith('Maximum humidity') :

                max_humidity = x.split(' ')[2]

                #print(max_humidity[:-1])

                if max_humidity == '' :

                        max_humidity = x.split(' ')[3]

                        print(max_humidity)

                max_humidity = max_humidity[:-1]

                days_record.append(max_humidity)

            if x.startswith('Minimum humidity') :

                min_humidity = x.split(' ')[2]

                if min_humidity == '' :

                        min_humidity = x.split(' ')[3]

                #print(min_humidity)

                min_humidity = min_humidity.split('%')[0]

                days_record.append(min_humidity)

            if x.startswith('Maximum pressure') :

                max_pressure = x.split(' ')[2]

                #print(max_pressure)

                days_record.append(max_pressure)

            if x.startswith('Minimum pressure') :

                min_pressure = x.split(' ')[2]

                #print(min_pressure)

                days_record.append(min_pressure)

            if x.startswith('Maximum windspeed') :

                max_windspeed = x.split(' ')[2]

                #print(max_windspeed)

                days_record.append(max_windspeed)

            if x.startswith('Maximum gust speed') :

                max_gust_speed = x.split(' ')[3]

                if max_gust_speed =='' :

                    max_gust_speed = x.split(' ')[4]

                    #print(max_gust_speed)

                days_record.append(max_gust_speed)

            if x.startswith('Maximum heat index') :

                max_heat_index = x.split(' ')[3]

                #print(max_heat_index[:-2])

                max_heat_index = max_heat_index[:-2]

                days_record.append(max_heat_index)

        month_data_list.append(days_record)

    return month_data_list

            
from bs4 import BeautifulSoup

import urllib.request

import time



month_text =['Jan', 'Feb' ,'Mar', 'Apr' ,'May' ,'Jun', 'Jul','Aug' ,'Sep','Oct','Nov','Dec']



for param in period[:] :

    #time.sleep(1)

    

    r =urllib.request.urlopen("https://www.estesparkweather.net/archive_reports.php?date=" + param)

    soup = BeautifulSoup(r,'lxml')

    data =[]

    all_table = soup.find_all('table')

    for table in all_table:

        data.append(table.text.splitlines())



    for x in data :

        while '' in x:

            x.remove('')



    final =[]

    total_data =[]

    index = param[4:6]

    int_index = int(index) -1

    

    for x in data:

        if x[0].startswith(month_text[int_index]) :

            final.append(x)

    print ( param + " : " + str(len(final)) )

    get_components(final , param , month_text[int_index])





if len(month_data_list) > 0 :

    print(len(month_data_list))



#print(month_data_list)



print("The End")

df = pd.DataFrame(month_data_list)

df.columns = ['Date','Average temperature (\xb0F)','Average humidity (%)','Average dewpoint (\xb0F)','Average barometer (in)','Average windspeed (mph)','Average gustspeed (mph)','Average direction (\xb0deg)','Rainfall for month (in)','Rainfall for year (in)', 'Maximum rain per minute','Maximum temperature (\xb0F)', 'Minimum temperature (\xb0F)','Maximum humidity (%)','Minimum humidity (%)','Maximum pressure','Minimum pressure','Maximum windspeed (mph)','Maximum gust speed (mph)','Maximum heat index (\xb0F)']

df1 = df[:-3]

df1['Date1'] =pd.to_datetime(df1['Date'])

df1.index = df1['Date1']

#df1.drop(['Date'] ,1 , inplace=True)

#df1 = df1.apply(pd.to_numeric, errors='ignore', downcast='float')



df1['Average temperature (°F)'] = df1['Average temperature (°F)'].astype(float)

df1['Average humidity (%)'] = df1['Average humidity (%)'].astype(float)

df1['Average dewpoint (°F)'] = df1['Average dewpoint (°F)'].astype(float)

df1['Average barometer (in)'] = df1['Average barometer (in)'].astype(float)

df1['Average windspeed (mph)'] = df1['Average windspeed (mph)'].astype(float)

df1['Average gustspeed (mph)'] = df1['Average gustspeed (mph)'].astype(float)

df1['Average direction (°deg)'] = df1['Average direction (°deg)'].astype(float)

df1['Rainfall for month (in)'] = df1['Rainfall for month (in)'].astype(float)

df1['Rainfall for year (in)'] = df1['Rainfall for year (in)'].astype(float)

df1['Maximum rain per minute'] = df1['Maximum rain per minute'].astype(float)

df1['Maximum temperature (°F)'] = df1['Maximum temperature (°F)'].astype(float)

df1['Minimum temperature (°F)'] = df1['Minimum temperature (°F)'].astype(float)

df1['Maximum pressure'] = df1['Maximum pressure'].astype(float)

df1['Minimum pressure'] = df1['Minimum pressure'].astype(float)

df1['Maximum windspeed (mph)'] = df1['Maximum windspeed (mph)'].astype(float)

df1['Maximum heat index (°F)'] = df1['Maximum heat index (°F)'].astype(float)

df1['Minimum humidity (%)'] = df1['Minimum humidity (%)'].astype(float)

df1['Maximum humidity (%)'] = df1['Maximum humidity (%)'].astype(float)

df1['Maximum gust speed (mph)'] = df1['Maximum gust speed (mph)'].astype(float)

from pandas.api.types import is_numeric_dtype, is_datetime64_dtype

df1.loc['2011-08-20' ,'Average windspeed (mph)']  =3.0







print(df1.info())



print(round(np.mean(df1["2011-08-1":"2011-08-20"]["Average windspeed (mph)"]), 2))

print(round(np.std(df1["2011-04-20":"2012-01-1"]["Minimum temperature (°F)"]), 2))

print(round(np.std(df1["2011-04-20":"2012-01-1"]["Maximum pressure"]), 2))

print(round(np.max(df1["2011-04-20":"2012-01-1"]["Maximum temperature (°F)"]), 2))



#print(is_numeric_dtype(df1['Maximum gust speed (mph)']))



print(str(len(df1.columns)))
'''['Average temperature (°F)', 'Average humidity (%)',

 'Average dewpoint (°F)', 'Average barometer (in)',

 'Average windspeed (mph)', 'Average gustspeed (mph)',

 'Average direction (°deg)', 'Rainfall for month (in)',

 'Rainfall for year (in)', 'Maximum rain per minute',

 'Maximum temperature (°F)', 'Minimum temperature (°F)',

 'Maximum humidity (%)', 'Minimum humidity (%)', 'Maximum pressure',

 'Minimum pressure', 'Maximum windspeed (mph)',

 'Maximum gust speed (mph)', 'Maximum heat index (°F)'] '''
df1['Month'] = df1['Date'].apply(lambda x:x.split('-')[1])

df_med_gust=df1['Maximum gust speed (mph)'].groupby(df1['Month']).median()

#print(df_med_gust)

print("Month with highest median = " +str(df_med_gust.idxmax())[:10])

print("higest median value = " + str(df_med_gust[str(df_med_gust.idxmax())[:10]]))
print("Average temparature from March 2010 to May 2012 = " + str(round(np.mean(df1['2010-03-01' : '2012-05-31']['Average temperature (°F)']),2)))
print("Std deviation of Max Wind speed =" + str(round(np.std(df1['Maximum windspeed (mph)']),2)))



p50_p75 = np.percentile(df1['Average temperature (°F)'],75) - np.percentile(df1['Average temperature (°F)'],50) 

print("Difference in 75 and 50 percentile for Average Temperature = " +str(round(p50_p75 , 2))) # 12.200000000000003

#print(p50_p75)



print("Pearson Coeff between Avg Dewpoint and Avg temparature = " +str(round(np.corrcoef(df1['Average dewpoint (°F)'], df1['Average temperature (°F)'])[0,1],2)))

#0.7596091253567396



np.min(df1['Average dewpoint (°F)'].groupby(pd.Grouper(freq="M")).mean()) #33.46666666666667



df_avg_humd = df1['Average dewpoint (°F)'].groupby(pd.Grouper(freq="M")).mean()

print("month with lowest average dew point = "+ str((df_avg_humd).idxmin())[:10])

#df1.loc[df1.groupby(pd.Grouper(freq="M"))['Average humidity (%)'].idxmin()]



print("Average temparature from March 2010 to May 2012 = " + str(round(np.mean(df1['2010-03-01' : '2012-05-31']['Average temperature (°F)']),2)))



df1['diff_pressure'] = df1['Maximum pressure'] -df1['Minimum pressure']

print("Day with highest difference in pressure = " + str(df1['diff_pressure'].idxmax())[:10])





df_med_gust=df1['Maximum gust speed (mph)'].groupby(pd.Grouper(freq='M')).median()

print("Month with highest median = " +str(df_med_gust.idxmax())[:10])

print("higest median value = " + str(df_med_gust[str(df_med_gust.idxmax())[:10]]))





#print(df1['2010-12-01':'2010-12-31']['Average temperature (°F)'].max())  #46.3

#print(df1['2010-12-01':'2010-12-31']['Average temperature (°F)'].min())  #1.5

print("Range of Temparature for Dec 2010 = " + str(df1['2010-12-01':'2010-12-31']['Average temperature (°F)'].max() - df1['2010-12-01':'2010-12-31']['Average temperature (°F)'].min()))



med_baro = np.median(df1['Average barometer (in)']) # 29.9

print("no of days less than median barometer reading = " + str(len(df1[df1['Average barometer (in)'] == med_baro])))



one_std_avg_temp = np.std(df1['Average temperature (°F)'])

mean_avg_temp = np.mean(df1['Average temperature (°F)'])



high = mean_avg_temp + one_std_avg_temp

low = mean_avg_temp - one_std_avg_temp



#print( one_std_avg_temp , mean_avg_temp , high ,low )

print("No of days with one Std deviation of average temp = " +str(len(df1[(df1['Average temperature (°F)'] > low ) & (df1['Average temperature (°F)'] < high)])))

#df1.info()
#print("Average temparature from March 2010 to May 2012 = " + str(round(np.mean(df1['2010-03-01' : '2012-02-28']['Average temperature (°F)']),2)))





df_med_gust=df1['Maximum gust speed (mph)'].groupby(pd.Grouper(freq='M')).median()

#print("Month with highest median = " +str(df_med_gust.idxmax())[:10])

#print("higest median value = " + str(df_med_gust[str(df_med_gust.idxmax())[:10]]))

#df1['2010-03-01' : '2012-05-31']['Average temperature (°F)' ]

df1[df1['Average temperature (°F)'] < 0]['Average temperature (°F)']

#print((df_med_gust.values))



df1.shape
df1.to_csv('climate_data.csv',index = False)