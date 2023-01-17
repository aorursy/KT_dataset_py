# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import package

import matplotlib.pyplot as plt

import seaborn as sns 

import statsmodels as sm

import folium as fl

from pathlib import Path

from sklearn.impute import SimpleImputer

import geopandas as gpd

import mapclassify as mpc

import warnings

from fbprophet import Prophet

from fbprophet.diagnostics import cross_validation

from fbprophet.diagnostics import performance_metrics

from statsmodels.tsa.stattools import grangercausalitytests

from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statsmodels.tsa.vector_ar.var_model import VAR

import plotly.offline as py

import plotly.express as px

import cufflinks as cf
%matplotlib inline

pd.options.plotting.backend

#pd.plotting.register_matplotlib_converters()

gpd.plotting.plot_linestring_collection

sns.set()

warnings.filterwarnings('ignore')
covidfile = '/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv'
covid19 = pd.read_csv(covidfile, parse_dates=True)
covid19.head(3)
covid19.isnull().sum()[covid19.isnull().sum()>0]
covid19.info()
covid19['ObservationDate'] = pd.DataFrame(covid19['ObservationDate'])

covid19['currentCase'] = covid19['Confirmed'] - covid19['Recovered'] - covid19['Deaths']
replace = ['Dem. Rep. Congo', "Côte d'Ivoire", 'Congo', 'United Kingdom', 'China','Central African Rep.',

          'Eq. Guinea','eSwatini','Bosnia and Herz.', 'S. Sudan', 'Dominican Rep.', 'W. Sahara',

          'United States of America']



name = ['Congo (Kinshasa)', 'Ivory Coast', 'Congo (Brazzaville)', 'UK', 'Mainland China', 

        'Central African Republic', 'Equatorial Guinea', 'Eswatini', 'Bosnia and Herzegovina', 'South Sudan',

       'Dominica', 'Western Sahara','US']
covid_data = covid19.drop(columns=['Province/State'])

covid_data = covid_data.replace(to_replace=name, value=replace)

# END Cleaning
covid_data.head()
gb_covid = covid_data.groupby('ObservationDate')[['Confirmed', 'Deaths', 'Recovered', 'currentCase']].agg('sum')

end_date = gb_covid.index.max()
print('========= COVID-19 Worldwide ==============================')

print("======== Report to date {} ===============\n".format(gb_covid.index.max()))

print('1- The number of country that are affected by COVID-19: {}'.format(len(covid_data['Country/Region'].unique())))

print('2- Total Confirmed: {}'.format(gb_covid['Confirmed'][gb_covid.index == gb_covid.index.max()].values[-1]))

print('3- Total Deaths: {}'.format(gb_covid['Deaths'][gb_covid.index==gb_covid.index.max()].values[-1]))

print('4- Total Recovered: {}'.format(gb_covid['Recovered'][gb_covid.index ==gb_covid.index.max()].values[-1]))

print('5- Total CurrentCase: {}'.format(gb_covid['currentCase'][gb_covid.index ==gb_covid.index.max()].values[-1]))

print('============================================================')
#plot the worldwide covid19

world_path_file = gpd.datasets.get_path('naturalearth_lowres') # upload natural data map

world = gpd.read_file(world_path_file)

world.head(3)
geo_merged = world.merge(covid_data[['ObservationDate','Country/Region','Confirmed','Deaths','Recovered','currentCase']] , 

                     left_on='name', right_on='Country/Region')
geo_merged.head(3)
geo_merged.info()
geo_merged['ObservationDate'] = pd.DataFrame(geo_merged['ObservationDate'])
geo_merged.plot(cmap='cividis_r', column='Confirmed', legend=True, figsize=(15,9), scheme='quantiles', k=6)

plt.title('SARS-Cov 2 in the worldwide')

plt.xlabel('Longitude')

plt.ylabel('Latitude')
worldwide = geo_merged.groupby(['ObservationDate','continent'])[['Confirmed','Deaths','Recovered','currentCase']].agg('sum').reset_index()
worldwide.head(3)
for c in worldwide.continent.unique():

    surface = worldwide[worldwide.continent==c]

    surface = surface.drop(columns='continent')

    surface.plot(x='ObservationDate',

    title='SARS Cov 2 confirmed, currentcase, recovered, deaths in {} continent over time'.format(c),

               figsize=(15,5))

    plt.ylabel('cummulative')
daily_case = geo_merged.loc[geo_merged.ObservationDate.isin([end_date])]
pop_size = daily_case.groupby('continent')['pop_est'].agg('sum')
case_size = daily_case.groupby('continent')['Confirmed'].agg('sum')
print('The number of positive case per population size in each continent at date: {} is:\n {}'.\

      format(end_date, case_size/pop_size))
def determinate_beta_gamma_delta(data=None):

    '''

        this function compute transmission rate, recovered rate and fatalities rate over time

        params: data

        return: beta, gamma, delta

    '''

    

    beta = []

    gamma = []

    delta = []

    

    for t in range(len(data.ObservationDate.values)):

        

        x = data.Confirmed.iloc[t]

        y = data.Deaths.iloc[t]

        z = data.Recovered.iloc[t]

        w = data.currentCase.iloc[t]

        

        if x == 0.0:

            beta.append(0)

            gamma.append(0)

            delta.append(0)

        else:

            beta_t = w/x

            gamma_t = z/x

            delta_t = y/x

            

            beta.append(beta_t)

            gamma.append(gamma_t)

            delta.append(delta_t)

            

    return np.array(beta), np.array(gamma), np.array(delta)        
geospatial = geo_merged.groupby(['ObservationDate','name','continent'])['Confirmed','Deaths','Recovered','currentCase'].agg('sum')
geospa = geospatial.reset_index()
geospa.head()
transmission, recovery, fatality = determinate_beta_gamma_delta(data=geospa)
geospa['beta'] = transmission

geospa['gamma'] = recovery

geospa['delta'] = fatality
geospa.head()
rate_map = geospa.groupby(['ObservationDate','continent'])[['beta','gamma','delta']].agg('mean').reset_index()
rate_map.head()
for c in rate_map.continent.unique():

    surface = rate_map[rate_map.continent==c]

    surface = surface.drop(columns='continent')

    surface.plot(x='ObservationDate',

    title='SARS Cov 2 transmission rate, recovery rate, delta rate in {} continent over time'.format(c),

                 figsize=(15,5))

    plt.ylabel('means rate')
worldwide[worldwide.continent=='Oceania'].plot(x='ObservationDate',  figsize=(15,5),

    title='SARS Cov 2 control in {} continent over time'.format('Oceania'),

                                              )

plt.ylabel('cummulative')
rate_map[rate_map.continent=='Oceania'].plot(x='ObservationDate', 

title='SARS Cov 2 transmission rate, recovery rate, delta rate in {} continent over time'.format('Oceania'),

                                                                        figsize=(15,5))

plt.ylabel('means rate')
oceania =  rate_map[rate_map.continent=='Oceania']
oceania['R0'] = oceania.beta.values/(oceania.gamma.values + oceania.delta.values)
oceania.plot(x='ObservationDate', y='R0',

             title='ratio reproduction number over time in Oceania',

              figsize=(15,5))

plt.ylabel('ratio')
africa = rate_map[rate_map.continent=='Africa']
africa ['R0'] = africa.beta.values/(africa.gamma.values + africa.delta.values)
africa.plot(x='ObservationDate', y='R0', 

             title='ratio reproduction number over time in Africa',

             figsize=(15,5))

plt.ylabel('ratio')
def find_R0(data=None):

    return data.beta.values/(data.gamma.values + data.delta.values)
china = geospa[geospa.name=='China']
china.head()
china[['ObservationDate','Confirmed','Deaths','Recovered','currentCase']].plot(x='ObservationDate', 

        title='SARS Cov 2 in China', figsize=(15,5))

plt.ylabel('Cummulative')
china[['ObservationDate','beta','gamma','delta']].plot(x='ObservationDate', 

                                                       title='SARS Cov 2 important parameters',

                                             figsize=(15,5))

plt.ylabel('rate')
# find R0

china['R0'] = find_R0(data=china)
china.plot(x='ObservationDate', y = 'R0', title='ratio reproductive number in China',

            figsize=(15,5))

plt.ylabel('ratio')
australia = geospa[geospa.name=='Australia']
australia.head()
australia[['ObservationDate','Confirmed','Deaths','Recovered','currentCase']].plot(x='ObservationDate', 

        title='SARS Cov 2 in Australia',  figsize=(15,5))

plt.ylabel('Cummulative')
australia[['ObservationDate','beta','gamma','delta']].plot(x='ObservationDate', 

                                                       title='SARS Cov 2 important parameters',

                                                        figsize=(15,5))

plt.ylabel('rate')
#compute R0

australia['R0'] = find_R0(data=australia)
australia.plot(x='ObservationDate', y = 'R0', title='ratio reproductive number in Australia',

                figsize=(15,5))

plt.ylabel('ratio')
cameroon = geospa[geospa.name=='Cameroon']
cameroon.head()
cameroon[['ObservationDate','Confirmed','Deaths','Recovered','currentCase']].plot(x='ObservationDate', 

        title='SARS Cov 2 in Cameroon', figsize=(15,5))

plt.ylabel('Cummulative')
cameroon[['ObservationDate','beta','gamma','delta']].plot(x='ObservationDate', 

                                                       title='SARS Cov 2 important parameters',

                                                       figsize=(15,5))

plt.ylabel('rate')
#Compute R0

cameroon['R0'] = find_R0(data=cameroon)
cameroon.plot(x='ObservationDate', y = 'R0',  title='ratio reproductive number in Cameroon',

               figsize=(15,5))

plt.ylabel('ratio')