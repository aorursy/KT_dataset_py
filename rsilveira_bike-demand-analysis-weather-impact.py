import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
#warnings.filterwarnings('ignore')
#Carregar dataset
#path = "data.csv"
path = "../input/data.csv"
df = pd.read_csv(path)
holidays = [
'2014-01-01','2014-12-25',
'2014-11-11','2014-07-04',
'2014-01-20','2014-02-17',
'2014-03-02','2014-05-26',
'2014-09-01','2014-10-13',
'2014-11-27','2015-01-01',
'2015-12-25','2015-11-11',
'2015-07-04','2015-01-19',
'2015-02-16','2015-03-02',
'2015-05-25','2015-09-07',
'2015-10-12','2015-11-26',
'2016-01-01','2016-12-25',
'2016-11-11','2016-07-04',
'2016-01-18','2016-02-15',
'2016-03-04','2016-05-30',
'2016-09-05','2016-10-10',
'2016-11-24','2017-01-01',
'2017-12-25','2017-11-11',
'2017-07-04','2017-01-16',
'2017-02-20','2017-03-06',
'2017-05-29','2017-09-04',
'2017-10-09','2017-11-23'
]

df_holidays = pd.DataFrame(holidays)
df_holidays['date'] = pd.to_datetime(df_holidays[0]).dt.date
df_holidays['holiday'] = 1
df_holidays = df_holidays.drop([0], axis=1)
#Corringindo campo day com 0
df.day =  df.starttime.astype(str).str.slice(8,10).astype(int)
#Starttime e Stoptime para data
df.starttime = pd.to_datetime(df.starttime)
df.stoptime  = pd.to_datetime(df.stoptime)
df['date']   = df.starttime.dt.date

#Definição fim de semana
df['weekend'] = np.where(df.starttime.dt.weekday > 4 , 1, 0)

#Inicializando novas features
df['season'] = ''

#Definição estações Hemisfério NORTE
for x in range(df.year.min().astype(int)-1, df.year.max().astype(int)+1):
    #Primavera: 21-03 até 20-06
    df.loc[(df.starttime >= str(x)+'-03-21 00:00:00') & (df.starttime <= str(x)+'-06-20 23:59:59'),  'season'] = 'primavera'
    #Verão:     21-06 até 20-09
    df.loc[(df.starttime >= str(x)+'-06-21 00:00:00') & (df.starttime <= str(x)+'-09-20 23:59:59'),  'season'] = 'verão'
    #Outono:    21-09 até 20-12
    df.loc[(df.starttime >= str(x)+'-09-21 00:00:00') & (df.starttime <= str(x)+'-12-20 23:59:59'),  'season'] = 'outono'
    #Inverno:   21-12 até 20-03
    df.loc[(df.starttime >= str(x)+'-12-21 00:00:00') & (df.starttime <= str(x+1)+'-03-20 23:59:59'),'season'] = 'inverno'
df_holidays.head()
df_join = df.set_index('date').join(df_holidays.set_index('date'))
df_join['isholiday']  = np.where(df_join.holiday==1, 1, 0)
df_join['regularday'] = np.where(df_join.isholiday+df_join.weekend == 0, 1, 0)
#clear DF
df.iloc[0:0]
del df
#Features Disponíveis
print(list(df_join))

#Data frame agregado
dfagg = df_join
df_map = df_join

#Coluna agregação viagens(trip_count)
dfagg['trip_count'] = 1

#Lista de features
col = ['date', 'year', 'month','day', 'hour','usertype', 'gender', 'events', 'weekend','isholiday', 'regularday', 'season']

#Agregação
dfagg =  dfagg.groupby(col).aggregate({  'temperature'     : 'mean',
                                         'tripduration'    : 'mean',
                                         'trip_count'      : 'sum',
                                         'dpcapacity_start': 'mean',
                                         'dpcapacity_end'  : 'mean'
                                        }).reset_index()

df_map =  df_map.groupby(['latitude_start','longitude_start']).aggregate({
                                         'trip_count'      : 'sum',
                                         'tripduration'    : 'mean'}).reset_index()
#clear DF_JOIN
df_join.iloc[0:0]
del df_join
dfagg.head(5)
dfagg_day = dfagg
conditions = [(dfagg_day['weekend'] == 1) ,
              (dfagg_day['isholiday'] == 1),
              (dfagg_day['regularday'] == 1)]

choices = ['WEEKEND', 'HOLIDAY',  'REGULAR']
dfagg_day['DayType'] = np.select(conditions, choices)
dfagg_day['CounDays'] = 1




#media final de semana ####
dfagg_hour_week = dfagg_day[dfagg_day.DayType=='WEEKEND']
dfagg_hour_week =  dfagg_hour_week.groupby(['date', 'hour']).aggregate({                                           
                                         'trip_count'      : 'sum'                                                                                  
                                        }).reset_index()
dfagg_hour_week =  dfagg_hour_week.groupby(['hour']).aggregate({                                           
                                         'trip_count'      : 'mean'                                                                                  
                                        }).reset_index()

#media dia de semana ####
dfagg_hour_reg = dfagg_day[dfagg_day.DayType=='REGULAR']
dfagg_hour_reg = dfagg_hour_reg.groupby(['date', 'hour']).aggregate({                                           
                                         'trip_count'      : 'sum'                                                                                  
                                        }).reset_index()
dfagg_hour_reg = dfagg_hour_reg.groupby(['hour']).aggregate({                                           
                                         'trip_count'      : 'mean'                                                                                  
                                        }).reset_index()

#media feriado ####
dfagg_hour_hol = dfagg_day[dfagg_day.DayType=='HOLIDAY']
dfagg_hour_hol = dfagg_hour_hol.groupby(['date', 'hour']).aggregate({                                           
                                         'trip_count'      : 'sum'                                                                                  
                                        }).reset_index()
dfagg_hour_hol = dfagg_hour_hol.groupby(['hour']).aggregate({                                           
                                         'trip_count'      : 'mean'                                                                                  
                                        }).reset_index()
#plot average travel chart
plt.rcParams["figure.figsize"] = (15,9)



plt.plot(dfagg_hour_week.hour, dfagg_hour_week.trip_count, label="Weekend")
plt.plot(dfagg_hour_reg.hour, dfagg_hour_reg.trip_count, label="Week")
plt.plot(dfagg_hour_hol.hour,dfagg_hour_hol.trip_count, label="Holiday")

# Add legend
plt.legend(loc='upper right')
plt.title("Average Travel by Type of Day (2014-2017)", fontsize=16, fontweight='bold')
plt.xlabel("Hour")
plt.ylabel("Average Travel per day")
plt.show()
#copy dataframe
dfagg_clima = dfagg
#Agregação
dfagg_clima =  dfagg_clima.groupby(['season', 'hour']).aggregate({                                          
                                         'tripduration'    : 'mean',
                                         'trip_count'      : 'sum'
                                        }).reset_index()
#Clear dfagg_clima 
del dfagg
dfagg_clima_verao = dfagg_clima[dfagg_clima.season=='verão']
dfagg_clima_prima = dfagg_clima[dfagg_clima.season=='primavera']
dfagg_clima_out   = dfagg_clima[dfagg_clima.season=='outono']
dfagg_clima_inver = dfagg_clima[dfagg_clima.season=='inverno']

plt.rcParams["figure.figsize"] = (15,9)

plt.plot(dfagg_clima_verao.hour, dfagg_clima_verao.trip_count, label="Summer")
plt.plot(dfagg_clima_prima.hour, dfagg_clima_prima.trip_count, label="Spring")
plt.plot(dfagg_clima_out.hour,dfagg_clima_out.trip_count, label="Autumn")
plt.plot(dfagg_clima_inver.hour,dfagg_clima_inver.trip_count, label="Winter")

# Add legend
plt.legend(loc='upper right')
plt.title("Travel by season (de 2014 até 2017)", fontsize=16, fontweight='bold')
plt.xlabel("Hour")
plt.ylabel("Travels")
plt.show()
dfagg_clima_verao = dfagg_clima[dfagg_clima.season=='verão']
dfagg_clima_inver = dfagg_clima[dfagg_clima.season=='inverno']
dfagg_clima_prima = dfagg_clima[dfagg_clima.season=='primavera']
dfagg_clima_out   = dfagg_clima[dfagg_clima.season=='outono']

plt.rcParams["figure.figsize"] = (15,9)

plt.plot(dfagg_clima_verao.hour, dfagg_clima_verao.tripduration, label="Summer")
plt.plot(dfagg_clima_inver.hour,dfagg_clima_inver.tripduration, label="Winter")
plt.plot(dfagg_clima_prima.hour, dfagg_clima_prima.tripduration, label="Spring")
plt.plot(dfagg_clima_out.hour,dfagg_clima_out.tripduration, label="Autumn")

# Add legend
plt.legend(loc='upper right')
plt.title("Average Travel (de 2014 até 2017)", fontsize=16, fontweight='bold')
plt.xlabel("Hour")
plt.ylabel("Average")
plt.show()