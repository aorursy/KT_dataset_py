import pandas as pd

import numpy as np

import requests

import datetime as dt
def get_weatherHistory_data(command, station, stname, datepicker, station_name):

    payload = {'command': command,

               'station': station,

               'stname': stname,

               'datepicker': datepicker}



    r = requests.get('http://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do', params = payload)

    tables = pd.read_html(r.text)

    df_temp = tables[1]

    df_temp.columns=df_temp.columns.droplevel(level=[0,1])



    df_temp.loc[0:23, 'ObsTime'] = datepicker +' '+ df_temp['ObsTime'].astype(str) +':00'

    df_temp.loc[23, 'ObsTime'] = (pd.to_datetime(datepicker)+dt.timedelta(days=1)).strftime('%Y-%m-%d') +' 0:00'



    df_temp['ObsTime'] = pd.to_datetime(df_temp['ObsTime'])

    df_temp.set_index('ObsTime',inplace=True)



    df_temp = df_temp.apply(pd.to_numeric, errors='coerce')

    df_temp.dropna(how='all', inplace=True)



    df_temp['station'] = station_name

    return df_temp
df_merged = pd.DataFrame()



starttime = '2020-07-01'

endtime = '2020-07-10'

delta_day = 1



timePeriod_temp = starttime



while timePeriod_temp <= endtime:

    try:

        df_temp = get_weatherHistory_data(command='viewMain',

                                          station='466920',

                                          stname='%E8%87%BA%E5%8C%97',

                                          datepicker=timePeriod_temp, 

                                          station_name='466920_臺北')

        timePeriod_temp = (pd.to_datetime(timePeriod_temp)+dt.timedelta(days=delta_day)).strftime('%Y-%m-%d')

        df_merged = pd.concat([df_merged, df_temp])         

    except:

        print('error')    

        

df_merged
df_merged.to_csv('Weather.csv')