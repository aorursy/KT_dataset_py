import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import datetime

warnings.filterwarnings('ignore')
Shanghai_composite= pd.read_csv("../input/000001.SS.csv")

Bombay_sensex= pd.read_csv("../input/^NSEI.csv")

Newyork_sp= pd.read_csv("../input/^GSPC.csv")

covid_19_data = pd.read_csv("../input/covid_19_data.csv")

confirmed = pd.read_csv("../input/time_series_covid_19_confirmed.csv")

deaths=pd.read_csv("../input/time_series_covid_19_deaths.csv")

recovered=pd.read_csv("../input/time_series_covid_19_recovered.csv")

#https://github.com/Kannavdhawan/Covid-19_an_economic_meltdown
Shanghai_composite['Date'].unique()
#updating with correct data 

confirmed.loc[confirmed['Country/Region'] == 'Italy', '3/12/20'] = 15113

recovered.loc[recovered['Country/Region'] == 'Italy', '3/12/20'] = 1258

deaths.loc[deaths['Country/Region'] == 'Italy', '3/12/20'] = 1016

confirmed.loc[confirmed['Province/State'] == 'United Kingdom', '3/12/20'] = 590

deaths.loc[deaths['Province/State'] == 'United Kingdom', '3/12/20'] = 10

recovered.loc[recovered['Province/State'] == 'United Kingdom', '3/12/20'] = 18
confirmed
col_names=list(confirmed)

dates=col_names[4:]
# confirmed[dates[1]]<confirmed[dates[0]]#return true false

#not using iterrows , not using normal comparison and then replacement.

# confirmed[dates[1]] = np.where(confirmed[dates[1]]<confirmed[dates[0]],confirmed[dates[0]],confirmed[dates[1]])



for i in range(len(dates)-1):

    confirmed[dates[i+1]]= np.where(confirmed[dates[i+1]]<confirmed[dates[i]],confirmed[dates[i]],confirmed[dates[i+1]])

    deaths[dates[i+1]]= np.where(deaths[dates[i+1]]<deaths[dates[i]],deaths[dates[i]],deaths[dates[i+1]])

    recovered[dates[i+1]]= np.where(recovered[dates[i+1]]<recovered[dates[i]],recovered[dates[i]],recovered[dates[i+1]])



    

# for index,row in confirmed.iterrows():

confirmed
#done 

# confirmed.iloc[347]

Shanghai_composite
#Simply reversed dataset .

Shanghai_composite=Shanghai_composite[::-1]

Shanghai_composite=Shanghai_composite.reset_index(drop=True)



Shanghai_composite



#Not working 



# Shanghai_composite=Shanghai_composite.sort_values(by="Date")

Shanghai_composite
# Bombay_sensex['Date']=pd.to_datetime(Bombay_sensex['Date'])\

#check unique dates to get more insights . it is swapping month and date locations.
# Bombay_sensex=Bombay_sensex.sort_values(by='Date',ascending=0)

Bombay_sensex=Bombay_sensex[::-1]

#removing 18th march which is extra .. 

# change it when updating the dataset

Bombay_sensex=Bombay_sensex.iloc[1:,:]

Bombay_sensex=Bombay_sensex.reset_index(drop=True)

#data was not uptodate

Bombay_sensex.iloc[0,-1]=21500

Bombay_sensex.head()
# Newyork_sp["Date"]=pd.to_datetime(Newyork_sp["Date"])

# Newyork_sp=Newyork_sp.sort_values(by='Date',ascending=0)

Newyork_sp=Newyork_sp[::-1]

Newyork_sp=Newyork_sp.reset_index(drop=True)

Newyork_sp.head()
#confirmed.info()
#converting wide to long 

confirmed=pd.melt(confirmed,id_vars=['Province/State','Country/Region', 'Lat', 'Long'],var_name='Date',

                                 value_name='Confirmed')

#for finding the fatality rate

deaths=pd.melt(deaths,id_vars=['Province/State','Country/Region', 'Lat', 'Long'],var_name='Date',

                value_name='Deaths')

recovered=pd.melt(recovered,id_vars=['Province/State','Country/Region', 'Lat', 'Long'],var_name='Date',

                  value_name='Recovered')

#value_vars= unspecified i.e using all the columns left.

#https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/

confirmed
deaths
recovered
#appending columns "death" and "recovered" from the two data frames to confirmed dataframe

confirmed['Deaths']=deaths['Deaths']

confirmed['Recovered']=recovered['Recovered']
confirmed
all_corona_data=confirmed[{'Country/Region','Date','Confirmed','Deaths','Recovered'}]
all_corona_data
# date_df=all_corona_data['Date']
Shanghai_composite
unique_dates=all_corona_data['Date'].unique()

unique_dates=unique_dates.tolist()



unique_countries=all_corona_data['Country/Region'].unique()

unique_countries=unique_countries.tolist()

full_df = pd.DataFrame(columns =['Country/Region','Deaths','Confirmed','Recovered'])



for date in unique_dates:

    datewise=all_corona_data.loc[all_corona_data['Date']==date,['Date','Country/Region','Deaths','Confirmed','Recovered']]     

    datewise=datewise.groupby(['Country/Region','Date']).sum().reset_index(drop=False)

    full_df=pd.concat([full_df,datewise])

# full_df['Date']=date_df

full_df=full_df.reset_index(drop=True)

full_df[::-1]
#testing 

# full_df.loc[full_df['Country/Region']=='Canada',:]

## test_df=pd.DataFrame({'a':[1,1,1,2,2],'b':[2,2,2,1,1]})

# test_df

# test_df.groupby(['b']).sum()
Shanghai_composite
#not working with differeent date formats 



# Shanghai_composite_after_outbreak=Shanghai_composite.loc[Shanghai_composite['Date']>'2020-01-21',:]

# Shanghai_composite_after_outbreak=Shanghai_composite_after_outbreak.sort_values(by='Date',ascending=1)

# Shanghai_composite_after_outbreak=Shanghai_composite_after_outbreak.reset_index(drop=True)



# Bombay_sensex_after_outbreak=Bombay_sensex.loc[Bombay_sensex['Date']>'2020-01-21',:]

# Bombay_sensex_after_outbreak=Bombay_sensex_after_outbreak.sort_values(by='Date',ascending=1)

# Bombay_sensex_after_outbreak=Bombay_sensex_after_outbreak.reset_index(drop=True)





# Newyork_sp_after_outbreak=Newyork_sp.loc[Newyork_sp['Date']>'2020-01-21',:]

# Newyork_sp_after_outbreak=Newyork_sp_after_outbreak.sort_values(by='Date',ascending=1)

# Newyork_sp_after_outbreak=Newyork_sp_after_outbreak.reset_index(drop=True)

index_of_outbreak_date=Shanghai_composite.loc[Shanghai_composite['Date']=='21-01-20'].index.values

index_of_outbreak_date=index_of_outbreak_date[0]



index_of_outbreak_date_1=Bombay_sensex.loc[Bombay_sensex['Date']=='21-01-20'].index.values

index_of_outbreak_date_1=index_of_outbreak_date_1[0]



index_of_outbreak_date_2=Newyork_sp.loc[Newyork_sp['Date']=='21-01-20'].index.values

index_of_outbreak_date_2=index_of_outbreak_date_2[0]
Shanghai_composite_after_outbreak=Shanghai_composite.iloc[0:index_of_outbreak_date,:]
Shanghai_composite_after_outbreak
Bombay_sensex_after_outbreak=Bombay_sensex.iloc[0:index_of_outbreak_date_1,:]
Bombay_sensex_after_outbreak
Newyork_sp_after_outbreak=Newyork_sp.iloc[0:index_of_outbreak_date_2,:]
Newyork_sp_after_outbreak
full_df.head()
india_corona=full_df.loc[full_df['Country/Region']=='India',:]

china_corona=full_df.loc[full_df['Country/Region']=='China',:]

US_corona=full_df.loc[full_df['Country/Region']=='US',:]
#so that i can replace the orig data with normalized data matching indexes

india_corona=india_corona.reset_index(drop=True)

china_corona=china_corona.reset_index(drop=True)

US_corona=US_corona.reset_index(drop=True)

india_corona.head()
import plotly.express as ax

fig = ax.bar(india_corona, x='Date', y='Confirmed', title='India: Confirmed Cases')

fig.show()
fig = ax.bar(china_corona, x='Date', y='Confirmed', title='China: Confirmed Cases')

fig.show()
fig = ax.bar(US_corona, x='Date', y='Confirmed', title='US: Confirmed Cases')

fig.show()
from sklearn.preprocessing import StandardScaler



india_corona_normalized=india_corona.copy()

china_corona_normalized=china_corona.copy()

US_corona_normalized=US_corona.copy()



scaler=StandardScaler()



y=scaler.fit(india_corona_normalized[['Confirmed','Deaths','Recovered']])

#using the same mean and std dev for other countries as well.



y=scaler.transform(india_corona_normalized[['Confirmed','Deaths','Recovered']])

x=pd.DataFrame(y)

india_corona_normalized[['Confirmed','Deaths','Recovered']]=x[[0,1,2]]



y=scaler.transform(china_corona_normalized[['Confirmed','Deaths','Recovered']])

x=pd.DataFrame(y)

china_corona_normalized[['Confirmed','Deaths','Recovered']]=x[[0,1,2]]



y=scaler.transform(US_corona_normalized[['Confirmed','Deaths','Recovered']])

x=pd.DataFrame(y)

US_corona_normalized[['Confirmed','Deaths','Recovered']]=x[[0,1,2]]
US_corona_normalized.head()
US_corona.head()
india_corona.head()
india_corona_normalized.head()
fig = ax.line(india_corona, x='Date', y='Confirmed', title='India: Confirmed Cases')

fig.show()
#fixing holidays for all indexes





china_corona
Shanghai_composite_after_outbreak
Bombay_sensex_after_outbreak.head()
Newyork_sp_after_outbreak.head()
fig = ax.line(Shanghai_composite_after_outbreak, x='Date', y='Close', title='Shanghai Composite')

fig.show()
fig = ax.line(Bombay_sensex_after_outbreak, x='Date', y='Close', title='Bombay Sensex')

fig.show()
fig = ax.line(Newyork_sp_after_outbreak, x='Date', y='Close', title='Newyork S & P 500')

fig.show()
# plt.figure(figsize=(10,10))

# plt.xticks(rotate=90) without axes 



#using axes object

fig,ax=plt.subplots(figsize=(13,8))

ax.plot(Shanghai_composite_after_outbreak.Date,Shanghai_composite_after_outbreak.Close,marker='o',color="green")

ax.set_xlabel("Dates")

ax.set_ylabel("Shanghai composite in yuan",color="green")

#rotating x axes ticks

# fig.autofmt_xdate() #or use for loop if not date 

for tick in ax.get_xticklabels():

    tick.set_rotation(90)



#in order to get two y axes on same plot  we use twinx()



ax2=ax.twinx()

ax2.plot(china_corona.Date,china_corona.Confirmed)



china_corona.head()