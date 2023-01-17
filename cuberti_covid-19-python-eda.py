import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt; from matplotlib.ticker import ScalarFormatter

import seaborn as sns; sns.set;

from datetime import timedelta

import os







from pdpbox import pdp

from plotnine import *

import feather



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

plt.style.use('fivethirtyeight') 
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv').rename(columns = {"Province_State": "State", "Country_Region":"country",

                                                                                                   "ConfirmedCases":"cases", "Fatalities":"deaths"})

train['Date']=pd.to_datetime(train.Date)

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

train['cou_state']=train.country +"-"+train.State.astype(str)
countries = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv')

countries['country'] = countries.country.str.strip()

#rename columns for easier coding

col_dict = {"Region":"Region",'Surface area (km2)':"surface_area", 'Population in thousands (2017)':'pop_in_thou', 

            'Population density (per km2, 2017)':'pop_density','Sex ratio (m per 100 f, 2017)':"sex_ratio",

            'GDP: Gross domestic product (million current US$)':'gdp','GDP growth rate (annual %, const. 2005 prices)':"gdp_growth", 

            'Urban population (% of total population)':'urban_pop','Urban population growth rate (average annual %)':'urban_growth_rate',

            'Fertility rate, total (live births per woman)':'fertility_rate','Life expectancy at birth (females/males, years)':'life_expectancy',

            'Population age distribution (0-14 / 60+ years, %)':'pop_age_distribution', 'International migrant stock (000/% of total pop.)':'imigrant_pop',

            'Infant mortality rate (per 1000 live births':'infant_mortality', 'Health: Total expenditure (% of GDP)':'healthcare_expenditure',

            'Health: Physicians (per 1000 pop.)':'physicians_per_1000'}

countries = countries.rename(columns = col_dict)

countries = countries[['country', 'surface_area', 'pop_in_thou','pop_density','sex_ratio','gdp','gdp_growth','urban_pop','urban_growth_rate', 'imigrant_pop',

           'fertility_rate','life_expectancy','pop_age_distribution','infant_mortality','healthcare_expenditure','physicians_per_1000']]

#Get the country names from the country data in-line with country nemes in COVID-19 Dataset

countries = countries.replace({'Bolivia (Plurinational State of)':'Bolivia', 'Congo':'Congo (Kinshasa)', 

                               'United States of America':'US', 'Iran (Islamic Republic of)':'Iran', 'Bosnia & Herzegovina': 'Bosnia and Herzegovina', 

                               'Venezuela (Bolivarian Republic of)':'Venezuela', "Viet Nam": "Vietnam", 'Trinidad & Tobago':'Trinidad and Tobago', 

                               'Republic of Korea':'Korea, South', 'Russian Federation':"Russia",'Brunei Darussalam':'Brunei',

                               'United Republic of Tanzania': 'Tanzania', 'Bahamas': 'The Bahamas', 'Gambia': 'Gambia, The',

                               'The former Yugoslav Republic of Macedonia': 'North Macedonia','Democratic Republic of the Congo': 'Republic of the Congo','Republic of Moldova':"Moldova",

                               'Taiwan': 'Taiwan*', 'Central African Rep.': 'Central African Republic', 'Czech Republic': 'Czechia'})

age_dist = countries.pop_age_distribution.str.split("/", expand = True)

countries['age_dist_0-14'] = age_dist[0].replace({"-99":np.nan, "...":np.nan}).astype(float)

countries['age_dist_60p'] = age_dist[1].replace({"-99":np.nan, "...":np.nan}).astype(float)

countries['age_dist_14-60'] = 100 - countries['age_dist_0-14']-countries['age_dist_60p']

countries = countries.drop(["pop_age_distribution", "surface_area", 'gdp_growth', 'urban_growth_rate'], axis = 1)

countries['gdp_per_cap']=(countries.gdp*1000) / (countries.pop_in_thou)

countries[countries.gdp_per_cap<0] = np.nan

life_ex = countries.life_expectancy.str.split("/", expand = True)

life_ex[0] = life_ex[0].replace({"-99":np.nan, "...":np.nan}).astype(float)

life_ex[1] = life_ex[1].replace({"-99":np.nan, "...":np.nan}).astype(float)

countries['life_expectancy'] = (life_ex[0]+life_ex[1])/2

countries['immigrand_pct'] = countries.imigrant_pop.str.split("/", expand = True, n=1)[1]

countries = countries.drop(['imigrant_pop', 'life_expectancy'], axis = 1)

countries = countries[~pd.isna(countries.country)]

countries['population'] = countries.pop_in_thou * 1000

#countries = countries.merge(train[['country', 'Lat', "Long"]].drop_duplicates(), how = 'left', on = 'country')

#data = Null;
def get_days_past_100(df, gr_col):

    #There is a lot of 'state' data by aggregating it you reduce the number of observations from 17k to 6.5k (on 3/23)

    data = df.groupby([gr_col, "Date"]).sum().reset_index()

    date_past_100 = data[data.cases > 100].groupby(gr_col)[['Date']].min().reset_index()[[gr_col,'Date']].rename(columns = {"Date":"date_past_100"})

    data = data.merge(date_past_100, how = 'left', on = gr_col)

    data['date_past_100'] = (data.date_past_100.fillna(pd.to_datetime(-999999)))

    data['days_since_100'] = ((data.Date - data.date_past_100)/(8.64e+13)).astype(int)

    data.loc[(data['days_since_100'] < 0) | (data['days_since_100'] > 10000), 'days_since_100'] = 0

    data = data.drop('date_past_100', axis = 1)

    return data



def add_country_data(df, countries):

    data_c = df.merge(countries, on = 'country', how = 'left')

    data_c['cases_p_1000'] = data_c.cases / (data_c.population / 1000)

    data_c['deaths_p_1000'] = data_c.deaths / (data_c.population / 1000)

    return data_c





def get_top_10(df, column):

    max_cases = df.groupby(column)[['cases_p_1000', 'deaths_p_1000', 'cases', 'deaths']].max().sort_values('cases_p_1000')

    top_10 = max_cases[~pd.isna(max_cases.cases_p_1000)].sort_values('cases', ascending = False).head(10).reset_index()

    #top_10_countries = top_10.country.tolist()

    top_10_percap = max_cases[~pd.isna(max_cases.cases_p_1000)].sort_values('cases_p_1000', ascending = False).head(10).reset_index()

    return top_10, top_10_percap
data_country = get_days_past_100(train, 'country')

data_state = get_days_past_100(train, 'cou_state').merge(train[['cou_state', 

                                                                'country', 'State']].drop_duplicates(), how = 'left', on = 'cou_state')

US_data = data_state.loc[data_state.country == 'US']
data_cou = add_country_data(data_country, countries)

data_us = add_country_data(US_data, countries)

data_state = add_country_data(data_state, countries)

top_10_cou, top_10_cou_percap = get_top_10(data_cou, 'country')

top_10_state, top_10_state_percap = get_top_10(data_us, 'State')
#Countries with most cases

top_10_cou
#countries with most cases per capita

top_10_cou_percap
def process_data(df, group_column):

    train_clean= pd.DataFrame([])

    for col in df[group_column].unique():

        cou_dat = df[df[group_column] == col].copy()

        cou_dat['pct_chg_cases']=(cou_dat.cases.pct_change()).fillna(0).tolist()

        cou_dat['days_to_double']=(1/cou_dat.cases.pct_change()).fillna(0).tolist()

        cou_dat['DtD_sma7'] = cou_dat.loc[:,'days_to_double'].rolling(window=7).mean()

        cou_dat['pct_c_sma7'] = cou_dat.loc[:,'pct_chg_cases'].rolling(window=7).mean()

        cou_dat['cases_10_days_prior'] = cou_dat[['cases']].shift(periods = 5).fillna(0)

        cou_dat['deaths_15_days_prior'] = cou_dat[['deaths']].shift(periods = 15).fillna(0)

        cou_dat['new_cases'] = cou_dat['cases'].diff().fillna(0)

        cou_dat['deaths_day'] = cou_dat['deaths'].diff().fillna(0)

        train_clean = train_clean.append(cou_dat)

    train_clean = train_clean.replace([np.inf, -np.inf], 0)

    return train_clean
data_cou = process_data(data_cou, 'country')

data_us = process_data(data_us, 'cou_state')

data_state = process_data(data_state, 'cou_state')
data_cou.columns
train.groupby('State').max().reset_index().country.unique()

#Countries that have state attributes towards them 

#These are provecnces that we can take into account when looking into when lockdowns were put into place
data_state.columns
train_raw = train.copy().drop('cou_state', axis = 1)

drop_col = ['cou_state', '']
def plot_case_growth(plot_dat, color_by, days_max = 35, ylim_daily = 10000, ylim_total = 100000):

    last_obs=plot_dat.groupby(color_by, as_index = False).max()

    f,ax = plt.subplots(1, 2, figsize=(20,7))

    #ax[0].set(yscale = 'log')

    #ax[0].set_ylim([100, ylim_daily])

    ax[0].set_xlim([0,days_max])

    sns.lineplot(x="days_since_100", y="new_cases", hue = color_by, data = plot_dat, markers = True, ax=ax[0])

    ax[0].set(xlabel='Days Since 100th Case', ylabel = "New Cases per Day", title = "New Daily Corona Cases")

    #plt.xticks(rotation=30)

    for i in range(0,last_obs.shape[0]):

        ax[0].text(min(last_obs.days_since_100.iloc[i], days_max), last_obs.new_cases.iloc[i], last_obs.loc[:,color_by].iloc[i])



    ax[1].set(yscale = 'log'); ax[1].set_ylim([100,ylim_total])

    ax[1].set_xlim([0,days_max])

    sns.lineplot(x="days_since_100", y="cases", hue = color_by, data = plot_dat, markers = True, ax=ax[1])

    ax[1].set(xlabel='Days Since 100th Case', ylabel = "Cases (log-scale)", title = "Cumulative Cases")

    ax[1].yaxis.set_major_formatter(ScalarFormatter())

    #plt.ticklabel_format(style='plain', axis='y')

    #plt.xticks(rotation=30)

    for i in range(0,last_obs.shape[0]):

        ax[1].text(min(last_obs.days_since_100.iloc[i], days_max), last_obs.cases.iloc[i], last_obs.loc[:,color_by].iloc[i])

        

        



def plot_per_cap_cases_and_deaths(plot_dat, color_by, days_max = 45, ylim_deaths = 20000):

    last_obs=plot_dat.groupby(color_by, as_index = False).max()

    f,ax = plt.subplots(1, 2, figsize=(20,7))

    ax[0].set(yscale = 'log')

    #ax[0].set_ylim([100,ylim_deaths])

    ax[0].set_xlim([0,days_max])

    sns.lineplot(x="days_since_100", y="cases_p_1000", hue = color_by, data = plot_dat, markers = True, ax=ax[0])

    ax[0].set(xlabel='Days Since 100th Case', ylabel = "Cases / 1000 country population", title = "Cases per 1000")

    #plt.xticks(rotation=30)

    for i in range(0,last_obs.shape[0]):

        ax[0].text(min(last_obs.days_since_100.iloc[i], days_max), last_obs.cases_p_1000.iloc[i], last_obs.loc[:,color_by].iloc[i])

    ax[1].set(yscale = 'log'); #ax[1].set_ylim([100,100000])

    ax[1].set_xlim([0,days_max])

    sns.lineplot(x="days_since_100", y="deaths_p_1000", hue = color_by, data = plot_dat, markers = True, ax=ax[1])

    ax[1].set(xlabel='Days Since 100th Case', ylabel = "Deaths / 1000 country population", title = "Deaths per 1000 Residents")

    ax[1].yaxis.set_major_formatter(ScalarFormatter())

    #plt.ticklabel_format(style='plain', axis='y')

    #plt.xticks(rotation=30)

    for i in range(0,last_obs.shape[0]):

        ax[1].text(min(last_obs.days_since_100.iloc[i], days_max), last_obs.deaths_p_1000.iloc[i], last_obs.loc[:,color_by].iloc[i])

        

def plot_deaths(plot_dat, color_by, days_max = 45, ylim_deaths = 20000):

    last_obs=plot_dat.groupby(color_by, as_index = False).max()

    f,ax = plt.subplots(1, 2, figsize=(20,7))

    ax[0].set(yscale = 'log')

    #ax[0].set_ylim([100,ylim_deaths])

    ax[0].set_xlim([0,days_max])

    sns.lineplot(x="days_since_100", y="deaths", hue = color_by, data = plot_dat, markers = True, ax=ax[0])

    ax[0].set(xlabel='Days Since 100th Case', ylabel = " Deaths", title = "Total Deaths")

    #plt.xticks(rotation=30)

    for i in range(0,last_obs.shape[0]):

        ax[0].text(min(last_obs.days_since_100.iloc[i], days_max), last_obs.deaths.iloc[i], last_obs.loc[:,color_by].iloc[i])

    #ax[1].set(yscale = 'log'); #ax[1].set_ylim([100,100000])

    ax[1].set_xlim([0,days_max])

    sns.scatterplot(x="days_since_100", y="deaths_day", size = 'deaths', hue = color_by, data = plot_dat, markers = True, ax=ax[1])

    ax[1].set(xlabel='Days Since 100th Case', ylabel = "Deaths / Day", title = "Daily Covid-19 Deaths")

    ax[1].yaxis.set_major_formatter(ScalarFormatter())

    #plt.ticklabel_format(style='plain', axis='y')

    #plt.xticks(rotation=30)

    for i in range(0,last_obs.shape[0]):

        ax[1].text(min(last_obs.days_since_100.iloc[i], days_max), last_obs.deaths_day.iloc[i], last_obs.loc[:,color_by].iloc[i])
plot_dat = data_cou[data_cou.country.isin(top_10_cou.country.unique())]

plot_case_growth(plot_dat, 'country', 45, 30000, 2000000)

plot_per_cap_cases_and_deaths(plot_dat, 'country')
plot_deaths(plot_dat, 'country')
f,ax = plt.subplots(figsize=(20,7))

last_obs=plot_dat.groupby('country', as_index = False).max()

#last_obs['cases'] = data.cases.astype(int)

#scale = 'cases',

sns.pointplot(x = 'cases_p_1000',y = 'deaths_p_1000', hue='country',  data = last_obs)

plt.xticks(rotation=30)

ax.set(yscale = 'log')

#ax.set(xscale = 'log')



for i in range(0,last_obs.shape[0]):

    ax.text((last_obs.cases_p_1000.iloc[i])*2, last_obs.deaths_p_1000.iloc[i], last_obs.country.iloc[i], horizontalalignment='left')
f,ax = plt.subplots(figsize=(20,7))

sns.lineplot(plot_dat.cases,plot_dat.deaths, hue=plot_dat.country)

ax.set(xlabel='Accumulated Cases', ylabel = "Accumulated Deaths", title = "Deaths vs Cases")

#ax.set(yscale = 'log')

#ax.set(xscale = 'log')



for i in range(0,last_obs.shape[0]):

    ax.text(last_obs.cases.iloc[i], last_obs.deaths.iloc[i], last_obs.country.iloc[i])
top_10_state.State
plot_dat = data_us[data_us.State.isin(top_10_state.State.unique().tolist())]

last_obs=plot_dat.groupby('State', as_index = False).max()

last_obs[['State', 'Date','cases', 'deaths']].sort_values('cases', ascending = False)
plot_case_growth(plot_dat, 'State')

plot_deaths(plot_dat, 'State')
plot_dat = data_us[data_us.State.isin(['Michigan', 'South Carolina'])]

plot_case_growth(plot_dat, 'State')

plot_deaths(plot_dat, 'State')

plot_per_cap_cases_and_deaths(plot_dat, 'State')