# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import pylab

import datetime as dt

from IPython.core.display import HTML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
corona_path = '../input/novel-corona-virus-2019-dataset/'

covid_19 = pd.read_csv(corona_path + 'covid_19_data.csv')

covid_19.drop(columns = ['SNo', 'Last Update', 'Recovered'], inplace = True)



column_name_map = {

    'Country/Region' : 'country',

    'Province/State' :  'state',

    'Confirmed' : 'cases',

    'Deaths' : 'deaths',

    'ObservationDate' : 'date',

}



covid_19.rename(columns = column_name_map, inplace = True)



country_name_map = {

    'US' : 'United States',

    'Mainland China' : 'China',

    'UK' : 'United Kingdom',

}

covid_19['country'] = covid_19['country'].replace(country_name_map)



covid_19_countries = covid_19.groupby(['country', 'date'], as_index = False).agg('sum')



covid_19_countries.head()
population_path = '../input/world-population-19602018/'

population = pd.read_csv(population_path + 'population_total_long.csv')



population.head()



population = population[population['Year']==2017].rename(columns = {'Country Name' : 'country', 'Count' : 'population'})[['country', 'population']]



country_name_map = {

    'Russian Federation' : 'Russia',

    'Slovak Republic' : 'Slovakia',

    'Egypt, Arab Rep.' : 'Egypt',

    'Korea, Rep.' : 'South Korea',

    'Iran, Islamic Rep.' : 'Iran',

}

population['country'] = population['country'].replace(country_name_map)



population.head()
latest_date = covid_19_countries['date'].max()

total_deaths_country = covid_19_countries[covid_19_countries['date']==latest_date][['country', 'deaths']]

total_deaths_country = total_deaths_country[total_deaths_country['deaths']!=0].sort_values(['deaths'], ascending = False)

total_deaths_country.head(10)
total_deaths_population = total_deaths_country.join(population.set_index('country'), on='country')

print(total_deaths_population[total_deaths_population['population'].isnull()])

total_deaths_population = total_deaths_population[total_deaths_population['population'].notnull() & (total_deaths_population['population'] >= 1e6)]
total_deaths_population['deaths_pm'] = total_deaths_population['deaths']/total_deaths_population['population']*1e6

total_deaths_population = total_deaths_population.sort_values(['deaths_pm'], ascending = False)

total_deaths_population.head()
fig = px.bar(total_deaths_population.head(20).sort_values(['deaths_pm'], ascending = True), x='deaths_pm', y='country', labels={'country':'Country', 'deaths_pm':'Deaths per Million'}, title='Deaths relative to Population', orientation='h', height=700)

fig.show()
daily_deaths_pop = covid_19_countries.join(population.set_index('country'), on='country')



latest_date = daily_deaths_pop['date'].max()

countries_with_deaths = list(daily_deaths_pop[(daily_deaths_pop['date']==latest_date) & (daily_deaths_pop['deaths']>0)]['country'].values)



daily_deaths_pop = daily_deaths_pop[daily_deaths_pop['country'].isin(countries_with_deaths)]

#print(daily_deaths_pop[daily_deaths_pop['population'].isnull()])

daily_deaths_pop = daily_deaths_pop[daily_deaths_pop['population'].notnull() & (daily_deaths_pop['population'] >= 1e6)]
daily_deaths_pop['Deaths per Million'] = daily_deaths_pop['deaths']/daily_deaths_pop['population']*1e6

top_countries = list(daily_deaths_pop[['country', 'Deaths per Million']].groupby(['country'], as_index=False).agg('max').sort_values('Deaths per Million', ascending=False).head(15)['country'].values)



daily_deaths_pop['Date'] = pd.to_datetime(daily_deaths_pop['date'])



fig = px.line(daily_deaths_pop[(daily_deaths_pop['country'].isin(top_countries)) & (daily_deaths_pop['Date'] > pd.Timestamp(2020,2,29))].sort_values(['Deaths per Million', 'Date'], ascending=False), x="Date", y="Deaths per Million", log_y=False, color='country', title='Deaths relative to population', height=600)

fig.show()
country_selection = [c for c in top_countries if c not in ['Italy', 'Spain', 'Belgium']]



fig = px.line(daily_deaths_pop[(daily_deaths_pop['country'].isin(country_selection)) & (daily_deaths_pop['Date'] > pd.Timestamp(2020,3,14))].sort_values(['Deaths per Million', 'Date'], ascending=False), x="Date", y="Deaths per Million", log_y=False, color='country', title='Deaths relative to population (excluding Belgium, Italy and Spain)', height=600)

fig.show()
deaths_population = daily_deaths_pop.pivot(columns = 'date', index='country', values = 'Deaths per Million').reset_index()



deaths_population.to_csv('deaths_population.csv',index=False)

deaths_population.head()
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1635677"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
covid_19_regions = covid_19[(covid_19['country']=='United States') | (covid_19['country']=='China')].copy(deep = True)

covid_19_regions['country'] = covid_19_regions['state'] + ' (' + covid_19_regions['country'] + ')'



covid_19_regions = pd.concat([covid_19_regions, covid_19[(covid_19['country']!='United States') & (covid_19['country']!='China')]])



covid_19_regions = covid_19_regions.groupby(['country', 'date'], as_index = False).agg('sum')



print(covid_19_regions['country'].unique())
region_path = '../input/us-states-and-regions-of-china-by-population/'



us_states = pd.read_csv(region_path + 'us_states_population.csv', thousands=',')

us_states = us_states.rename(columns = {'State' : 'country', 'Population estimate, July 1, 2019[2]' : 'population'})[['country', 'population']]

us_states['country'] = us_states['country'] + ' (United States)'

us_states.head()



china_regions = pd.read_csv(region_path + 'china_regions_population.csv', thousands=',')

china_regions = china_regions.rename(columns = {'Administrative Division' : 'country', '2017' : 'population'})[['country', 'population']]

china_regions['country'] = china_regions['country'] + ' (China)'

china_regions.head()



population_regions = pd.concat([population, us_states, china_regions])

#print(population_regions['country'].unique())
daily_regions = covid_19_regions.join(population_regions.set_index('country'), on='country')



latest_date = daily_regions['date'].max()

regions_with_deaths = list(daily_regions[(daily_regions['date']==latest_date) & (daily_regions['deaths']>0)]['country'].values)



daily_regions = daily_regions[daily_regions['country'].isin(regions_with_deaths)]

#print(daily_regions[daily_regions['population'].isnull()])

daily_regions = daily_regions[daily_regions['population'].notnull() & (daily_regions['population'] >= 1e6)]
daily_regions['Deaths per Million'] = daily_regions['deaths']/daily_regions['population']*1e6

top_regions = list(daily_regions[['country', 'Deaths per Million']].groupby(['country'], as_index=False).agg('max').sort_values('Deaths per Million', ascending=False).head(15)['country'].values)



daily_regions['Date'] = pd.to_datetime(daily_regions['date'])



fig = px.line(daily_regions[(daily_regions['country'].isin(top_regions)) & (daily_regions['Date'] >= pd.Timestamp(2020,3,1))].sort_values(['Deaths per Million', 'Date'], ascending=False), x="Date", y="Deaths per Million", log_y=False, color='country', title='Deaths relative to population', height=600)

fig.show()
for co in top_regions:

    cases = np.array(daily_regions[daily_regions['country']==co]['cases'])

    deaths = np.array(daily_regions[daily_regions['country']==co]['deaths'])

    

    mind = 1

    maxd = min(15, len(cases)-5)

    

    n = len(cases)-maxd

    d = deaths[maxd:]



    delay_est = []



    for delay in range(mind, maxd+1):

        c = cases[:-delay]

        c = c[-n:]



        fac = np.dot(d, c)/np.dot(c, c)

        residuals = fac*c-d

        ss_res = np.sum(residuals**2)

        ss_tot = np.sum((d - np.mean(d))**2)

        rs = 1 - (ss_res / ss_tot)



        delay_est.append((rs, delay, fac))



    delay_est.sort(reverse=True)

    

    dopt = delay_est[0][1]

    popt = delay_est[0][2]

    

    deaths_est = np.concatenate([[0] * dopt, popt * cases])

    

    pylab.plot(range(len(deaths_est)), deaths_est)

    pylab.plot(range(len(deaths)), deaths)

    pylab.show()



    print(co, delay_est[0])
df_countries = []

df_deaths = []

df_dates = []



latest_date = pd.to_datetime(daily_regions['date'].max())



for co in top_regions:

    cases = np.array(daily_regions[daily_regions['country']==co]['cases'])

    deaths = np.array(daily_regions[daily_regions['country']==co]['deaths'])

    

    mind = 1

    maxd = min(15, len(cases)-5)

    

    n = len(cases)-maxd

    d = deaths[maxd:]



    delay = 6

    

    c = cases[:-delay]

    c = c[-n:]



    fac = np.dot(d, c)/np.dot(c, c)

    residuals = fac*c-d

    ss_res = np.sum(residuals**2)

    ss_tot = np.sum((d - np.mean(d))**2)

    rs = 1 - (ss_res / ss_tot)

    

    deaths_est = np.concatenate([[0] * delay, fac * cases])

    

    df_countries.extend([co] * delay)

    df_deaths.extend(deaths_est[-delay:])

    df_dates.extend([latest_date + dt.timedelta(days=i) for i in range(1, delay+1)])

    

    print(co, rs, fac)

    

    pylab.plot(range(len(deaths_est)), deaths_est)

    pylab.plot(range(len(deaths)), deaths)

    pylab.show()

    

print(df_countries)

print(df_deaths)

print(df_dates)
data = {'country':  df_countries,

        'deaths' : df_deaths,

        'Date' : df_dates,

        }



daily_est = pd.DataFrame(data, columns = data.keys())



daily_est = daily_est.join(population_regions.set_index('country'), on='country')



daily_est['Deaths per Million'] = daily_est['deaths']/daily_est['population']*1e6



daily_est.head()
daily_total = pd.concat([daily_regions[['country', 'deaths', 'Date', 'population', 'Deaths per Million']], daily_est])



plot_regions = [r for r in top_regions if r not in []]



fig = px.line(daily_total[(daily_total['country'].isin(plot_regions)) & (daily_total['Date'] > pd.Timestamp(2020,2,29))].sort_values(['Date', 'Deaths per Million'], ascending=False), x="Date", y="Deaths per Million", log_y=False, color='country', title='Deaths relative to population', height=600)

fig.show()
daily_total = pd.concat([daily_regions[['country', 'deaths', 'Date', 'population', 'Deaths per Million']], daily_est])



plot_regions = [r for r in top_regions if r not in ['New York (United States)']]



fig = px.line(daily_total[(daily_total['country'].isin(plot_regions)) & (daily_total['Date'] > pd.Timestamp(2020,3,15))].sort_values(['Date', 'Deaths per Million'], ascending=False), x="Date", y="Deaths per Million", log_y=False, color='country', title='Deaths relative to population (excluding New York)', height=600)

fig.show()