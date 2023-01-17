import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import warnings

warnings.filterwarnings('ignore')



df_city_day = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')

df_city_hour = pd.read_csv('../input/air-quality-data-in-india/city_hour.csv')

df_station_day = pd.read_csv('../input/air-quality-data-in-india/station_day.csv')

df_station_hour = pd.read_csv('../input/air-quality-data-in-india/station_hour.csv')

df_stations = pd.read_csv('../input/air-quality-data-in-india/stations.csv')



list_of_df = [df_city_day,df_city_hour,df_station_day,df_station_hour,df_stations]

list_of_df_name = ['df_city_day','df_city_hour','df_station_day','df_station_hour','df_stations']

print(f"Available datasets are: {list_of_df_name}")

for i,df in zip(list_of_df_name,list_of_df):

    print(f"Columns of {i} are \n{df.columns}\n")
df_city_day.head()
#creating a func to make missing value table so that it can be used again

def missing_value_table(df):

    values = df.isnull().sum()

    percentage = 100*df.isnull().sum()/len(df)

    table = pd.concat([values,percentage.round(2)],axis=1)

    table.columns = ['No of missing values','% of missing values']

    return table[table['No of missing values']!=0].sort_values('% of missing values',ascending=False).style.background_gradient('Greens')

    

missing_value_table(df_city_day)
#converting dtype of date column to datetime

df_city_day['Date']=df_city_day['Date'].apply(pd.to_datetime)

#setting date column as index

df_city_day.set_index('Date',inplace=True)
#            imputing missing values using interpolation

#df_city_day.interpolate(method='linear',axis=0,limit_direction='both',inplace=True)



#            imputing AQI_Bucket column according to AQI column.

#def custom_imputer(df):

 #   if df['AQI'] < 51.0:

  #      return 'Good'

   # elif 50.0<df['AQI']<101.0:

    #    return 'Satisfactory'

#    elif 100.0<df['AQI']<201.0:

 #       return 'Moderate'

  #  elif 200.0<df['AQI']<301.0:

   #     return 'Poor'

    #elif 300.0<df['AQI']<401.0:

#        return 'Very Poor'

 #   else:

  #      return 'Severe'



#df_city_day['AQI_Bucket'] = df_city_day.apply(custom_imputer,axis=1)
print(f"City data is available from {df_city_day.index.min().date()} to {df_city_day.index.max().date()}")
df_city_day[['City','AQI']].groupby('City').mean().sort_values('AQI').plot(kind='barh',cmap='summer',figsize=(8,8))

plt.title('Average AQI in last 5 years');
city_day = df_city_day.copy()

city_day['BTX'] = city_day['Benzene']+city_day['Toluene']+city_day['Xylene']

city_day['Particulate_Matter'] = city_day['PM2.5']+city_day['PM10']

city_day['Nitrogen Oxides'] = city_day['NO']+city_day['NO2']+city_day['NOx']

city_day.drop(['Benzene','Toluene','Xylene','PM2.5','PM10','NO','NO2','NOx'],axis=1,inplace=True)



plt.figure(figsize=(5,4))

sns.heatmap(city_day.corr(),cmap='coolwarm',annot=True);
pollutants = ['City','AQI_Bucket', 'AQI', 'Particulate_Matter', 'Nitrogen Oxides','NH3', 'CO', 'SO2', 'O3',  'BTX']

city_day = city_day[pollutants]

print('Distribution of different pollutants in last 5 years')

city_day.plot(kind='line',figsize=(18,18),cmap='coolwarm',subplots=True,fontsize=10);
def max_polluted_cities(pollutant):

    table = city_day[[pollutant,'City']].groupby(["City"]).mean().sort_values(by=pollutant,ascending=False).reset_index()

    return table[:5].style.background_gradient(cmap='Reds')



print("Cities having worst levelss of each pollutant-")

for pollutant in pollutants[2:]:

    df = max_polluted_cities(pollutant)

    display(df)
city_ahmedabad = city_day[city_day['City']=='Ahmedabad']

city_ahmedabad['month']=city_ahmedabad.index.month

city_ahmedabad['year']=city_ahmedabad.index.year

print("AQI distribution in Ahmedabad")

fig,axes=plt.subplots(1,2,figsize=(10,5))

sns.pointplot(x='month',y='AQI',data=city_ahmedabad,ax=axes[0])

sns.pointplot(x='year',y='AQI',data=city_ahmedabad,ax=axes[1]);
#extracting date from df_city_hour

df_city_hour['Datetime'] = df_city_hour['Datetime'].apply(pd.to_datetime)

df_city_hour['Hour'] = df_city_hour['Datetime'].apply(lambda x: x.hour)



city_ahmedabad_hour = df_city_hour[df_city_hour['City']=='Ahmedabad']

sns.pointplot(x='Hour',y='AQI',data=city_ahmedabad_hour,color='Orange')

plt.title('AQI level throughout the day in Ahmedabad');
city_Delhi = city_day[city_day['City']=='Delhi']

city_Delhi['month']=city_Delhi.index.month

city_Delhi['year']=city_Delhi.index.year

print("AQI distribution in Delhi")

fig,axes=plt.subplots(1,2,figsize=(10,5))

sns.pointplot(x='month',y='AQI',data=city_Delhi,ax=axes[0],color='Green')

sns.pointplot(x='year',y='AQI',data=city_Delhi,ax=axes[1],color='Green');
city_delhi_hour = df_city_hour[df_city_hour['City']=='Delhi']

sns.pointplot(x='Hour',y='AQI',data=city_delhi_hour,color='Orange')

plt.title('AQI level throughout the day in Delhi');
city_Patna = city_day[city_day['City']=='Patna']

city_Patna['month']=city_Patna.index.month

city_Patna['year']=city_Patna.index.year

print("AQI distribution in Patna")

fig,axes=plt.subplots(1,2,figsize=(10,5))

sns.pointplot(x='month',y='AQI',data=city_Patna,ax=axes[0],color='Purple')

sns.pointplot(x='year',y='AQI',data=city_Patna,ax=axes[1],color='Purple');
city_Patna_hour = df_city_hour[df_city_hour['City']=='Patna']

sns.pointplot(x='Hour',y='AQI',data=city_Patna_hour,color='Orange')

plt.title('AQI level throughout the day in Patna');