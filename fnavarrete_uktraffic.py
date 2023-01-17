# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import folium

import geopandas as gdp

from folium.plugins import HeatMap

from datetime import datetime

import statsmodels.api as sm

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
pd_2005_2007 = pd.read_csv("../input/accidents_2005_to_2007.csv")

pd_2009_2011 = pd.read_csv("../input/accidents_2009_to_2011.csv")

pd_2012_2014 = pd.read_csv("../input/accidents_2012_to_2014.csv")

pd_2005_2007.columns
pd_2005_2007.columns.values
df_concat1 = pd_2005_2007.append(pd_2009_2011)
df_uk = pd_2012_2014.append(df_concat1)

df_uk.head()
df_uk.dtypes
df_uk.describe()
len(df_uk['Time'])
df_uk.isnull().values.any()
df_uk.isnull().sum()
df_uk.hist(column='Urban_or_Rural_Area',figsize=(10,10))
plt.figure(figsize=(10,10))

df_uk['Road_Type'].groupby(df_uk['Road_Type']).count().plot(kind="bar")
pd_tmp = df_uk.iloc[:30000,:]
max_amount = float(pd_tmp['Number_of_Casualties'].max())



hmap = folium.Map(location=[54, -3.20], zoom_start=5, )



HeatMap( list(zip(pd_tmp.Latitude.values, pd_tmp.Longitude.values, pd_tmp.Number_of_Casualties.values)),

                   min_opacity=0.2,

                   max_val=max_amount*2,

                   radius=17, blur=15, 

                   max_zoom=1, 

                 ).add_to(hmap)
hmap
def full_date(row):

    try:

        newdate = datetime.strptime(row['Date'] + ' ' + str(row['Time']), '%d/%m/%Y %H:%M') 

    except:

        row['Time'] = '00:00'

        newdate = datetime.strptime(row['Date'] + ' ' + str(row['Time']), '%d/%m/%Y %H:%M')

    return newdate
df_uk['FullDate'] = df_uk.apply(lambda row: full_date(row),axis=1)

df_uk['datetime'] = pd.to_datetime(df_uk['FullDate'])
plt.figure(figsize=(10,10))

df_uk['FullDate'].groupby(df_uk['FullDate'].dt.hour).count().plot(kind="bar")
df = df_uk.set_index('datetime')

df.drop(['Date'], axis=1, inplace=True)



df.head()

df_years = df['FullDate'].groupby([df.index.year,df.index.month,df.Accident_Severity]).count()

df_years = df_years.reset_index(level=0, inplace=False)



df_years = df_years.rename(index=str, columns={'datetime':"year"})



df_years = df_years.reset_index(level=0, inplace=False)



df_years = df_years.rename(index=str, columns={'datetime':"month"})



df_years
df_years = df_years.reset_index(level=0, inplace=False)
df_pivot = pd.pivot_table(df_years, index=['month'],columns=['Accident_Severity','year'], values=['FullDate'])

df_pivot
df_pivot['FullDate']['3'].plot(figsize=(10,10),title='Slight')
df_pivot['FullDate']['2'].plot(figsize=(10,10),title='serious')
df_pivot['FullDate']['1'].plot(figsize=(10,10),title='Fatal')
df['Accident_Severity'].groupby(df['Accident_Severity']).count().plot(kind="bar")
cause_columns = ['Speed_limit','Pedestrian_Crossing-Human_Control','Pedestrian_Crossing-Physical_Facilities'

                  ,'Light_Conditions','Weather_Conditions','Road_Surface_Conditions','Special_Conditions_at_Site'

                 , 'Carriageway_Hazards']
plt.figure(figsize=(15,25))

counter = 0

for i in cause_columns:

    plt.subplot(8,1,counter+1)

    plt.xticks([])

    df_uk[i].groupby(df_uk[i]).count().plot(kind="bar")

    counter = counter + 1

plt.tight_layout()

plt.show()
df_per_year = df['FullDate'].groupby(df.index.year).count()

X = df_per_year.index.values

Y = df_per_year.values
lambda_poiss = np.sum(Y)/len(Y)

lambda_poiss
df_ym = df_years

df_ym = df_ym.sort_index()
def date_ym(row):

    newdate = datetime.strptime(str(row['year']) + ' ' + str(row['month']), '%Y %m') 

    return newdate
df_ym['date_ym'] = df_ym.apply(lambda row: date_ym(row),axis=1)

df_ym.set_index('date_ym',inplace=True)

df_ym.rename(columns={'FullDate':'NumAccidents'},inplace=True)
severity = ['Fatal','Serious','Slight']
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



counter = 0

Accident_Severity = ['1','2','3']



plt.figure(figsize=(15,25))



for i in Accident_Severity:



    df_train = df_ym[df_ym['Accident_Severity'] == i]['2005':'2013']

    df_test = df_ym[df_ym['Accident_Severity'] == i]['2014']

    

    #df_train = df_train.sort_index()

    #df_test = df_test.sort_index()



    y_hat_avg_tmp = df_test.copy()

    fit1 = ExponentialSmoothing(np.asarray(df_train['NumAccidents']) ,seasonal_periods=12 ,trend='add', seasonal='add',).fit()

    y_hat_avg_tmp['Holt_Winter'] = fit1.forecast(len(df_test['NumAccidents']))

    

    if counter == 0:

        y_hat_avg = y_hat_avg_tmp.copy()

    else:

        y_hat_avg = y_hat_avg.append(y_hat_avg_tmp)

    

    plt.subplot(3,1,counter+1)

    plt.title(severity[counter])

    plt.plot( df_train['NumAccidents'], label='Train')

    plt.plot(df_test['NumAccidents'], label='Test')

    plt.plot(y_hat_avg_tmp['Holt_Winter'], label='Holt_Winter')

    plt.legend(loc='best')



    counter = counter + 1    

    

plt.show()
df_data = df_ym.groupby(['year','month']).sum()

df_data.reset_index(inplace=True)

df_data['date_ym'] = df_data.apply(lambda row: date_ym(row),axis=1)

df_data['datetime'] = pd.to_datetime(df_data['date_ym'])

df_data.set_index('datetime',inplace=True)

df_data= df_data.sort_index()
df_forecast=y_hat_avg.groupby(['year','month']).sum()

df_forecast.reset_index(inplace=True)

df_forecast['date_ym'] = df_forecast.apply(lambda row: date_ym(row),axis=1)

df_forecast['datetime'] = pd.to_datetime(df_forecast['date_ym'])

df_forecast.set_index('datetime',inplace=True)

df_forecast= df_forecast.sort_index()
plt.figure(figsize=(16,8))

#plt.plot( df_train['NumAccidents'], label='Train')

plt.plot(df_data['NumAccidents'], label='Data')

plt.plot(df_forecast['Holt_Winter'], label='Holt_Winter')

#plt.plot(df_forecast['NumAccidents'], label='Data')



plt.legend(loc='best')

plt.show()
from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(df_forecast.NumAccidents, df_forecast.Holt_Winter))

print(rms)