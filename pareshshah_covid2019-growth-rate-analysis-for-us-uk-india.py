import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Any results you write to the current directory are saved as output.
%matplotlib inline 

import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df= pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
df.shape
df.info()
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df['Last Update'] = pd.to_datetime(df['Last Update'])
df['Confirmed']=df['Confirmed'].astype('int')
df['Deaths']=df['Deaths'].astype('int')
df['Recovered']=df['Recovered'].astype('int')

from datetime import date
#df_update=df.loc[df.ObservationDate>pd.Timestamp(date(2020,4,12))]
df_update=df
df_update
df_update.isnull().sum()
df_update['Province/State']=df_update.apply(lambda x: x['Country/Region'] if pd.isnull(x['Province/State']) else x['Province/State'],axis=1)
df['Province/State']=df.apply(lambda x: x['Country/Region'] if pd.isnull(x['Province/State']) else x['Province/State'],axis=1)
df_update['Country/Region']=df_update.apply(lambda x:'China' if x['Country/Region']=='Mainland China' else x['Country/Region'],axis=1)
df['Country/Region']=df.apply(lambda x:'China' if x['Country/Region']=='Mainland China' else x['Country/Region'],axis=1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_update['ProvinceID'] = le.fit_transform(df_update['Province/State'])
df_update['CountryID']=le.fit_transform(df_update['Country/Region'])
df_update.head()
print(df['Country/Region'].unique())
print("\nNumber of unique countries in the dataset",len(df['Country/Region'].unique()))
new_df=df.groupby(['Country/Region','ObservationDate'],as_index=False).sum()
new_df['NewCases']=0
new_df['GrowthRate']=0

country=new_df.loc[0]['Country/Region']
new_df['NewCases'][0]=new_df.loc[0]['Confirmed']
for i in range(1,len(new_df)):
    new_country=new_df.loc[i]['Country/Region']
    if(country == new_country):
        prev_day_total=new_df.loc[i-1]['Confirmed']
        today_total=new_df.loc[i]['Confirmed']
        if(today_total > 0):
            new_cases=today_total-prev_day_total
            daily_growth_rate = np.round((new_cases / today_total) * 100,2)
    else:
        new_cases=new_df['Confirmed'][i]
        daily_growth_rate = 0
        country = new_country
    new_df['NewCases'][i]=new_cases
    new_df['GrowthRate'][i]=daily_growth_rate
        

from datetime import date 
from datetime import timedelta 
today = date.today() 
print("Today is: ", today) 
  
# Yesterday date 
yesterday = today - timedelta(days = 1) 
print("Yesterday was: ", yesterday) 

top10_df=new_df.loc[new_df['ObservationDate'] == yesterday].sort_values('Confirmed',ascending=False).head(10)
top10_country=top10_df['Country/Region'].values
print('Top 10 Countries:',top10_country)

next15_df=new_df.loc[new_df['ObservationDate'] == yesterday].sort_values('Confirmed',ascending=False).head(25).tail(15)
next15_country=next15_df['Country/Region'].values
print('Next Top 15 Countries:',next15_country)
new_df.loc[new_df['Country/Region'].isin(['India','US','UK','Russia'])][['Country/Region','ObservationDate','NewCases' ]]\
    .pivot(index='ObservationDate',values='NewCases',columns='Country/Region')\
    .plot(kind='line',figsize=(30,10),title='New Cases Growth Trend for US, UK, Russia and India')

new_df.loc[new_df['Country/Region'].isin(['US','UK','India'])][['Country/Region','ObservationDate','GrowthRate' ]]\
    .pivot(index='ObservationDate',values='GrowthRate',columns='Country/Region')\
    .plot(kind='line',figsize=(30,10),title='New Cases Growth Rate for US, UK, Russia and India')


new_df.loc[new_df['Country/Region'].isin(np.delete(top10_country,0))][['Country/Region','ObservationDate','Confirmed' ]]\
    .pivot(index='ObservationDate',values='Confirmed',columns='Country/Region')\
    .plot(kind='line',figsize=(20,10),title='New Cases Growth Trend for Top 10 Countries',marker='+',ylim=(0,500000),yticks=(1,10000,100000))

new_df.loc[new_df['Country/Region'].isin(next15_country)][['Country/Region','ObservationDate','NewCases' ]]\
    .pivot(index='ObservationDate',values='NewCases',columns='Country/Region')\
    .plot(kind='line',figsize=(30,20),title='New Cases Growth Trend for Next Top 15 Countries')

new_df.loc[new_df['Country/Region'].isin(top10_country)][['Country/Region','ObservationDate','GrowthRate' ]]\
    .pivot(index='ObservationDate',values='GrowthRate',columns='Country/Region')\
    .plot(kind='line',figsize=(30,20),title='New Cases Growth Trend for Top 10 Countries')

