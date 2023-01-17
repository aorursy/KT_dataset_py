import pandas as pd 
import math
import sys
import warnings

warnings.simplefilter("ignore")

# read data
df_conf_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_reco_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
df_dead_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

# make a copy
df_conf = df_conf_raw
df_reco = df_reco_raw
df_dead = df_dead_raw

#print(df_conf['Country/Region'].sort_values().tolist())

# filter to germany
df_conf = df_conf[df_conf['Country/Region']=='Germany']
df_reco = df_reco[df_reco['Country/Region']=='Germany']
df_dead = df_dead[df_dead['Country/Region']=='Germany']

# add label
df_reco['Label'] = 'Recovered'
df_conf['Label'] = 'Confirmed'
df_dead['Label'] = 'Deaths'

# transpose and union
df = pd.concat([df_conf,df_reco,df_dead])
df = df.drop(['Province/State','Country/Region','Lat','Long'],1)
df_transposed = df.transpose()
cols = df_transposed.loc[['Label']].values.tolist()[0]
ts = df.drop('Label',1).transpose()
ts.columns = cols
ts.index = pd.to_datetime(ts.index)

# add absolute metrics
ts['Active'] = ts.Confirmed - ts.Recovered - ts.Deaths
ts['NewConfirmed'] = ts.Confirmed - ts.Confirmed.shift(periods=1)
ts['NewRecovered'] = ts.Recovered - ts.Recovered.shift(periods=1)
ts['NewDeaths'] = ts.Deaths - ts.Deaths.shift(periods=1)

# add relative metrics
ts['NewRecoveredRate'] = ts['NewRecovered'] / ts['NewConfirmed']
ts['DeathRate'] = ts['Deaths'] / ts['Confirmed']
ts['RecoveredRate'] = ts['Recovered'] / ts['Confirmed']
ts['NewConfirmedRate'] = ts['NewConfirmed'] / (ts['Confirmed'] - ts['NewConfirmed'] )
import matplotlib.pyplot as plt

ts_filtered = ts[ts['Confirmed'] > 500]
#ts_filtered.plot()
#print(ts_filtered.columns)

#fig, axs = plt.subplots(2, 2, figsize=(15,8))

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,6))

ts_filtered[['Confirmed','Recovered','Active']].plot(ax=axes[0,0])
ts_filtered[['NewConfirmed','NewRecovered','Deaths','NewDeaths']].plot(ax=axes[1,0])
ts_filtered[['NewRecoveredRate','RecoveredRate','NewConfirmedRate']].plot(ax=axes[0,1]) 
ts_filtered[['DeathRate']].plot(ax=axes[1][1])

#fig.show()
pd.set_option("display.max_rows", 30)
ts[ts.Confirmed>0].sort_index(ascending=False).head(30)
country_list = ['Germany','Italy','Spain','Austria','Portugal','France','Denmark','Switzerland','United Kingdom','Netherlands']
country_df = {}

for country in country_list:
    df_conf = df_conf_raw
    df_reco = df_reco_raw
    df_dead = df_dead_raw
    df_conf = df_conf[df_conf['Country/Region']==country].groupby('Country/Region').sum()
    df_reco = df_reco[df_reco['Country/Region']==country].groupby('Country/Region').sum()
    df_dead = df_dead[df_dead['Country/Region']==country].groupby('Country/Region').sum()
    
    # add label
    df_reco.loc[:,'Label'] = 'Recovered'
    df_conf.loc[:,'Label'] = 'Confirmed'
    df_dead.loc[:,'Label'] = 'Deaths'
    # transpose and union
    df = pd.concat([df_conf,df_reco,df_dead])
    #df = df.drop(['Province/State','Lat','Long','Country/Region'],1)
    df = df.drop(['Lat','Long'],1)
    df_transposed = df.transpose()
    cols = df_transposed.loc[['Label']].values.tolist()[0]
    ts = df.drop('Label',1).transpose()
    ts.columns = cols
    ts.index = pd.to_datetime(ts.index)

    # add absolute some metrics
    ts['Active'] = ts.Confirmed - ts.Recovered - ts.Deaths
    ts['NewConfirmed'] = ts.Confirmed - ts.Confirmed.shift(periods=1)
    country_df[country] = ts[ts.index > '2020-03-01']
    
cols = 3
rows = math.ceil(len(country_df)/cols) 
fig, axes = plt.subplots(nrows=rows, ncols=cols,figsize=(21,3*rows))
i = 0
for country in country_df:
    country_df[country][['Confirmed','Active','Recovered']].plot(title=country,ax=axes[int(math.floor(i/cols)),i%cols])
    i+=1
def moving_average (df,n):
    data2 = df.rolling(n).mean()
    col_list = data2.columns.tolist()
    new_cols = []
    for i in col_list:
        new_cols += [str(i) + '_ma' + str(n)]
    data2.columns = new_cols 
    return data2

"""
fig2, axes2 = plt.subplots(nrows=rows, ncols=cols,figsize=(21,3*rows))
i = 0
for country in country_df:
    data = country_df[country][['NewConfirmed']][country_df[country]['NewConfirmed'].shift(periods=-2) > 0]
    #moving_average(data,3).plot(title=country,ax=axes2[int(math.floor(i/cols)),i%cols])
    i+=1
"""
data = moving_average(pd.concat(country_df),3).NewConfirmed_ma3
#data
data.unstack(level=0).plot(figsize=(21,9),title='New confirmed cases by country with moving average 3')
population_data = [
    ['Germany',82500000],
    ['France',67000000],
    ['United Kingdom',65800000],
    ['Italy',60600000],
    ['Spain',46500000],
    ['Netherlands',17100000],
    ['Portugal',10300000],
    ['Austria',8700000],
    ['Switzerland',8400000],
    ['Denmark',5750000]
]
population_df = pd.DataFrame(population_data,columns=['country','population'])
population_df = population_df.set_index('country')


for country in country_list:
    mrc = country_df[country].sort_index(ascending=False).head(1).Confirmed[0]
    population_df.loc[country,'most_recent_confirmed'] = mrc
    
population_df['Infection_Rate'] = population_df.most_recent_confirmed / population_df.population

population_df.Infection_Rate.sort_values(ascending=False).plot(kind='bar')

import numpy as np
population_df['confirmed_per_mil'] = (population_df.Infection_Rate*1000000).astype(int)
population_df.sort_values('confirmed_per_mil',ascending=False)
    