import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



%matplotlib inline



sns.set(style='darkgrid', palette='pastel', color_codes=True)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (16, 6)



def dty(X,s=5):

    return display (X.shape, pd.concat([pd.DataFrame(X.dtypes).T, X.head(s)],axis=0) )

time_series_covid_19_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

dty(time_series_covid_19_deaths)



time_series_covid_19_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

dty(time_series_covid_19_recovered)



time_series_covid_19_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

dty(time_series_covid_19_confirmed)
drop = ['Province/State','Lat','Long']



def tr(x):

  return x.drop(drop,axis=1).groupby('Country/Region').sum()



time_series_covid_19_deaths = tr(time_series_covid_19_deaths)

time_series_covid_19_recovered = tr(time_series_covid_19_recovered)

time_series_covid_19_confirmed = tr(time_series_covid_19_confirmed)
# combine time series



keys = ['confirmed','deaths','recovered',]

time_series = pd.concat([time_series_covid_19_confirmed,time_series_covid_19_deaths,time_series_covid_19_recovered,],

          axis=0,keys=keys,names=['status'])



time_series
time_series = time_series.T

time_series.index = pd.to_datetime(time_series.index)



df = pd.DataFrame(time_series.unstack(level=0)).reset_index()

df = df.rename(columns={'level_2':'date',0:'cases'})
g = sns.lineplot(data=df.groupby(['status','date']).sum().reset_index(),x='date', y='cases',style='status')

g.set_title('COVID-19 worldwide spread')

plt.show()
g = sns.lineplot(data=df[df['status']=='confirmed'], x='date', y='cases',hue='Country/Region', style="status")

plt.legend(

    loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=5,

    handleheight=2.4, labelspacing=0.05)

g.set_title('COVID-19 worldwide spread in particular countries')

plt.show()



# this graph is not presentable. let's look at the particular countries
cols = [

        'Province/State','Country/Region',

        'Lat','Long','3/28/20']

df2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')[cols]

#pd.read_csv('time_series_covid_19_recovered.csv')

#pd.read_csv('time_series_covid_19_confirmed.csv')

dty(df2)
df2['cases at location'] = df2['3/28/20'].astype(str) + '\n' +df2['Country/Region']+' \n'+df2['Province/State'].fillna('')

df2['r'] = np.where(df2['Province/State'].fillna('nann')=='nann',1,0)

df2['r'] = df2['r']*np.log(1+df2['3/28/20']) + (1-df2['r'])*(1+np.log(df2['3/28/20']))

# let's check the map

import folium



Map = folium.Map(location=(55.751244, 37.618423), zoom_start=1, width=800, height=500)

colors = {1 : 'red', 0 : 'blue'}



df2.apply(

    lambda row: folium.CircleMarker(

        location=[row['Lat'], row['Long']],

        popup=str(row['cases at location']),

        radius=row['r'], 

        #color=colors[row['burned']],

        #fill_color=colors[row['burned']],

        fill=True,

        ).add_to(Map), axis=1)



Map
# let's check Russia and China 



fig, ax = plt.subplots(figsize=(20,12))

for i,j in enumerate(['Russia','China']):

  plt.subplot(2,1,1+i)

  #what_to_plot = [(x,y) for x in ['confirmed','deaths','recovered'] for y in [j]]

  #g = sns.lineplot(data=time_series.T.loc[what_to_plot].T)

  g = sns.lineplot(data=df[df['Country/Region']==j], 

             x='date', y='cases',

             hue='Country/Region', style="status")

  g.set_title('COVID-19 spread in '+j)

  #plt.legend(what_to_plot, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3, handleheight=2.4, labelspacing=0.05)

plt.show()