

import numpy as np

import pandas as pd

import seaborn as sns

from datetime import datetime
corona= pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#after the loading the data. Next step is to view/see the top 10 rows of the loaded data set



corona.head(10)
#last 10 rows of loaded data set



corona.tail(10)
corona.describe()
#information about each var



corona.info()
#we will be listing the columns of all the data.

#we will check all columns



corona.columns
corona.sample(frac=0.01)
#sample: random rows in the dataset

#useful for future analysis

corona.sample(5)
#next, how many rows an columns are there in the loaded data set



corona.shape
# and, will check null on all the data and if there is any null, getting the sum of all the null data's



corona.isna().sum()
df= corona.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum()

df=df.reset_index()

df=df.sort_values('ObservationDate', ascending= True)

df.head(60)
df= corona.groupby('Province/State')['Confirmed','Deaths','Recovered'].sum()

df=df.reset_index()

df=df.sort_values('Province/State', ascending= True)

df.head(60)
#df=corona[corona['Confirmed'] == corona['Deaths']+['Recovered']]

#df=df[['Province','Confirmed','Deaths','Recovered']]#

#df=df.reset_index()

#df=df.sort_values('Confirmed',ascending= True)

#df.head()
df= corona.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum()

df.sort_values('ObservationDate',ascending=True)

df.head(10)
print(min(corona.Confirmed))

print(max(corona.Confirmed))

print(corona.Confirmed.mean())
print(min(corona.Deaths))

print(max(corona.Deaths))

print(corona.Deaths.mean())
print(min(corona.Recovered))

print(max(corona.Recovered))

print(corona.Recovered.mean())
#loading the raw data of confirmed, deaths and confirmed



conf=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

death=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recov=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")



print(conf.shape)

print(death.shape)

print(recov.shape)



conf.head()
conf2 = pd.melt(conf, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

death2 = pd.melt(death, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])

recov2 = pd.melt(recov, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name=['Date'])



print(conf2.shape)

print(death2.shape)

print(recov2.shape)



conf2.head()
# Converting the new column to dates



conf2['Date'] = pd.to_datetime(conf2['Date'])

death2['Date'] = pd.to_datetime(death2['Date'])

recov2['Date'] = pd.to_datetime(recov2['Date'])

#renaming the values to confirmed,death and recivered with respected datasets



conf2.columns=conf2.columns.str.replace('value','Confirmed')

death2.columns=death2.columns.str.replace('value','Deaths')

recov2.columns=recov2.columns.str.replace('value','Recovered')
#Finding the sum of NaN values in the columns of respective loaded data set



print(conf2.isna().sum())

#print(death2.isna().sum())

#print(recov2.isna().sum())
#Dealing with the Nan values



conf2['Province/State'].fillna(conf2['Country/Region'], inplace=True)

death2['Province/State'].fillna(death2['Country/Region'], inplace=True)

recov2['Province/State'].fillna(recov2['Country/Region'], inplace=True)



conf2.isna().sum()
print(conf2.shape)

print(death2.shape)

print(recov2.shape)
join= conf2.merge(death2[['Province/State','Country/Region','Date','Deaths']], 

                                      how = 'outer', 

                                      left_on = ['Province/State','Country/Region','Date'], 

                                      right_on = ['Province/State', 'Country/Region','Date'])



join2= join.merge(recov2[['Province/State','Country/Region','Date','Recovered']], 

                                      how = 'outer', 

                                      left_on = ['Province/State','Country/Region','Date'], 

                                      right_on = ['Province/State', 'Country/Region','Date'])



join2.head()



df= join2.groupby('Country/Region')['Confirmed','Deaths','Recovered'].sum()

df=df.reset_index()

df=df.sort_values('Country/Region', ascending= True)

df.head(60)
#Verifying is there any null values from the above data



join2.isna().sum()
#Adding month and year as a new column



join2['Month-Year'] = join2['Date'].dt.strftime('%b-%Y')
join2.head(10)
df= join2.groupby('Month-Year')['Confirmed','Deaths','Recovered'].sum()

df.sort_values('Month-Year',ascending=True)

df.head()