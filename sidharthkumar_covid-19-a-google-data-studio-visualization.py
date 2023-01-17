from IPython.display import IFrame

IFrame('https://datastudio.google.com/embed/reporting/bdd0af88-f4df-4357-a67b-550f9e7ad9c0/page/QQaIB', width='100%', height=900)
#basic libraries

import pandas as pd

import numpy as np

import os

import gc

import datetime as dt

import math

from IPython.display import Image

import warnings

warnings.filterwarnings('ignore')
#Directly pulling data from the source

path1 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"

path2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"

path3 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"

data_crfm = pd.read_csv(path1)

data_dead = pd.read_csv(path2)

data_reco = pd.read_csv(path3)
data_crfm.head()
#What I am doing here is aloting the previous value to next date if the cumulative count is less on next date!

def modifier(x):

    return(x[0] if x[0]>x[1] else x[1])



def data_correctr(data):

    total_cols = data.shape[1]

    cols = data.columns

    for i in range(5,total_cols):

        data[cols[i]] = data[[cols[i-1], cols[i]]].apply(modifier, 1)

    return data
#getting corrected data set!

data_crfm_c = data_correctr(data_crfm)

data_dead_c = data_correctr(data_dead)

data_reco_c = data_correctr(data_reco)
total_cols = data_crfm_c.shape[1]



data_crfm_d = data_crfm_c.copy()

data_dead_d = data_dead_c.copy()

data_reco_d = data_reco_c.copy()



# this is done to calculate the percentage for every day (initalising day 1 to zero)

data_crfm_p = data_crfm_c.copy()

data_crfm_p.iloc[:,4] = 0

data_dead_p = data_dead_c.copy()

data_dead_p.iloc[:,4] = 0

data_reco_p = data_reco_c.copy()

data_reco_p.iloc[:,4] = 0





for i in range(5,total_cols):

    

    #converting cumulative to daily count

    data_crfm_d.iloc[:, i] = data_crfm_d.iloc[:, i] - data_crfm_c.iloc[:, i-1]

    data_dead_d.iloc[:, i] = data_dead_d.iloc[:, i] - data_dead_c.iloc[:, i-1]

    data_reco_d.iloc[:, i] = data_reco_d.iloc[:, i] - data_reco_c.iloc[:, i-1]

    

    #percentage change: I will store the previous day cumulative and apply percentage change later

    data_crfm_p.iloc[:, i] = data_crfm_c.iloc[:, i-1]

    data_dead_p.iloc[:, i] = data_dead_c.iloc[:, i-1]

    data_reco_p.iloc[:, i] = data_reco_c.iloc[:, i-1]



# Here I am storing previous day daily count I will need this to calculate percentage change metric: the 6 small box in the dashboard

data_crfm_dp = data_crfm_d.copy()  

data_crfm_dp.iloc[:,4] = 0

data_dead_dp = data_dead_d.copy()

data_dead_dp.iloc[:,4] = 0

data_reco_dp = data_reco_d.copy()

data_reco_dp.iloc[:,4] = 0



for i in range(5,total_cols):

    #percentage change: I will store the previous day daily and apply percentage change later

    data_crfm_dp.iloc[:, i] = data_crfm_d.iloc[:, i-1]

    data_dead_dp.iloc[:, i] = data_dead_d.iloc[:, i-1]

    data_reco_dp.iloc[:, i] = data_reco_d.iloc[:, i-1]

# Here comes the melt funtion of pandas. One line and your coloumns turns into rows!

df_crfm = pd.melt(data_crfm_d, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Daily Confirmed"})

df_dead = pd.melt(data_dead_d, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Daily Death"})

df_reco = pd.melt(data_reco_d, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Daily Recovered"})



df_crfm_c = pd.melt(data_crfm_c, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Cum Confirmed"})

df_dead_c = pd.melt(data_dead_c, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Cum Death"})

df_reco_c = pd.melt(data_reco_c, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Cum Recovered"})



df_crfm_p = pd.melt(data_crfm_p, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"PCum Confirmed"})

df_dead_p = pd.melt(data_dead_p, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"PCum Death"})

df_reco_p = pd.melt(data_reco_p, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"PCum Recovered"})



df_crfm_dp = pd.melt(data_crfm_dp, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"dPCum Confirmed"})

df_dead_dp = pd.melt(data_dead_dp, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"dPCum Death"})

df_reco_dp = pd.melt(data_reco_dp, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"dPCum Recovered"})
df_crfm.head()
print(df_crfm.shape, df_dead.shape, df_reco.shape, df_crfm_c.shape, df_dead_c.shape, df_reco_c.shape, df_crfm_p.shape, df_dead_p.shape, df_reco_p.shape)
#Collecting the metric into 1 data frame

df = df_crfm.merge(df_dead[['Country/Region','Lat','Long', 'Time', 'Daily Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_reco[['Country/Region','Lat','Long', 'Time', 'Daily Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])



df = df.merge(df_crfm_c[['Country/Region','Lat','Long', 'Time', 'Cum Confirmed']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_dead_c[['Country/Region','Lat','Long', 'Time', 'Cum Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_reco_c[['Country/Region','Lat','Long', 'Time', 'Cum Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])



df = df.merge(df_crfm_p[['Country/Region','Lat','Long', 'Time', 'PCum Confirmed']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_dead_p[['Country/Region','Lat','Long', 'Time', 'PCum Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_reco_p[['Country/Region','Lat','Long', 'Time', 'PCum Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])



df = df.merge(df_crfm_dp[['Country/Region','Lat','Long', 'Time', 'dPCum Confirmed']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_dead_dp[['Country/Region','Lat','Long', 'Time', 'dPCum Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_reco_dp[['Country/Region','Lat','Long', 'Time', 'dPCum Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df.head()
#Some metrics



df['Mortality'] = ((df['Cum Death'] *100) /df['Cum Confirmed']).replace([np.inf, -np.inf], np.nan).fillna(0)

df['perchange Confirmed'] = ((df['Daily Confirmed']*100)/df['PCum Confirmed']).replace([np.inf, -np.inf], np.nan).fillna(0)

df['perchange Death'] = ((df['Daily Death']*100)/df['PCum Death']).replace([np.inf, -np.inf], np.nan).fillna(0)

df['perchange Recovered'] = ((df['Daily Recovered']*100)/df['PCum Recovered']).replace([np.inf, -np.inf], np.nan).fillna(0)
#last datatype corrections before feeding to data studio

df['Time'] = pd.to_datetime(df['Time']).dt.date

df['Lat Long'] = df['Lat'].astype(str)+","+df['Long'].astype(str)
df.to_csv("C:\\Users\\sikumar\\Downloads\\Corona_virus.csv", index = False)

print("Done!")
Image("../input/some-images-datastudio/Create new data source.PNG")
Image("../input/some-images-datastudio/Create new data source 2.PNG")
Image("../input/some-images-datastudio/Dtypes.PNG")
Image("../input/some-images-datastudio/workspace.PNG")