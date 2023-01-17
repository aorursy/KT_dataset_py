import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from itertools import cycle, islice
import seaborn as sb
import matplotlib.dates as dates
import datetime as dt

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly import tools, subplots
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from pathlib import Path
import pandas as pd
pop_dir = Path('../input/countryinfo/covid19countryinfo.csv')
pop_data = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')
pdata = pop_data[["country", "pop", "density"]]
pdata =pdata.rename(columns={'country':'Country_Region', 'pop':'Population'})

# convert columns population from String to float, to be able to divide
pdata['Population'] = pdata['Population'].str.replace(',', '')
pdata['Population'] = pdata['Population'].astype(float)

updata = pd.DataFrame(pdata.groupby('Country_Region')['Population', 'density'].max()).reset_index()
# Korea, South
updata.head()

updata['Country_Region'] = updata['Country_Region'].map({'US': 'United States', 
                                                         'Korea, South': 'South Korea'}).fillna(updata['Country_Region'])
print(updata[updata['Country_Region']=='South Korea'])
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv", parse_dates=['Date'])#index_col=0
display(train_data.head())
# display(train_data.dtypes)
test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv", parse_dates=['Date'])#index_col=0
display(test_data.head())

# train_data[train_data['Country_Region']== 'US']
sum_df = pd.pivot_table(train_data, values=['ConfirmedCases','Fatalities'], index=['Date'],aggfunc=np.sum)
display(sum_df.max())
train_data['NewConfirmedCases'] = train_data['ConfirmedCases'] - train_data['ConfirmedCases'].shift(1)
train_data['NewConfirmedCases'] = train_data['NewConfirmedCases'].fillna(0.0)
train_data['NewFatalities']     = train_data['Fatalities'] - train_data['Fatalities'].shift(1)
train_data['NewFatalities']     = train_data['NewFatalities'].fillna(0.0)#.astype(int)
train_data['MortalityRate']     = train_data['Fatalities'] / train_data['ConfirmedCases']
train_data['MortalityRate']     = train_data['MortalityRate'].fillna(0.0)
train_data['GrowthRate']        = train_data['NewConfirmedCases']/train_data['NewConfirmedCases'].shift(1)
train_data['GrowthRate']        = train_data['GrowthRate'].replace([-np.inf, np.inf],  0.0)
train_data['GrowthRate']        = train_data['GrowthRate'].fillna(0.0) 

def getColumnInfo(df):
    n_province =  df['Province_State'].nunique()
    n_country  =  df['Country_Region'].nunique()
    n_days     =  df['Date'].nunique()
    start_date =  df['Date'].unique()[0]
    end_date   =  df['Date'].unique()[-1]
    return n_province, n_country, n_days, start_date, end_date

n_train = train_data.shape[0]
n_test = test_data.shape[0]

n_prov_train, n_count_train, n_train_days, start_date_train, end_date_train = getColumnInfo(train_data)
n_prov_test,  n_count_test,  n_test_days,  start_date_test,  end_date_test  = getColumnInfo(test_data)

# print ('<==Train data==> \n # of Province_State: '+str(n_prov_train),', # of Country_Region:'+str(n_count_train), 
#        ', Time Period: '+str(start_date_train)+' to '+str(end_date_train), '==> days:',str(n_train_days))
# print("\n Countries with Province/State information:  ", train_data[train_data['Province_State'].isna()==False]['Country_Region'].unique())
# print ('\n <==Test  data==> \n # of Province_State: '+str(n_prov_test),', # of Country_Region:'+str(n_count_test),
#        ', Time Period: '+start_date_test+' to '+end_date_test, '==> days:',n_test_days)

df_test = test_data.loc[test_data['Date'] > '2020-04-14']
overlap_days = n_test_days - df_test.Date.nunique()
print('\n overlap days with training data: ', overlap_days, ', total days: ', n_train_days+n_test_days-overlap_days)

prob_confirm_check_train = train_data.ConfirmedCases.value_counts(normalize=True)
prob_fatal_check_train = train_data.Fatalities.value_counts(normalize=True)

n_confirm_train = train_data.ConfirmedCases.value_counts()[1:].sum()
n_fatal_train = train_data.Fatalities.value_counts()[1:].sum()

print('Percentage of confirmed case records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_confirm_train, n_train, prob_confirm_check_train[1:].sum()*100))
print('Percentage of fatality records = {0:<2.0f}/{1:<2.0f} = {2:<2.1f}%'.format(n_fatal_train, n_train, prob_fatal_check_train[1:].sum()*100))
train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',
                                                                                         'GrowthRate':'last' })

#display(train_data_by_country.tail(10))
max_train_date = train_data['Date'].max()
train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)


#
train_data_by_country_confirm = train_data_by_country_confirm.merge(updata, on ="Country_Region", how='left')

train_data_by_country_confirm['InfectedPopulation_Ratio'] = train_data_by_country_confirm['ConfirmedCases'].div(train_data_by_country_confirm['Population'],fill_value=0)*100
train_data_by_country_confirm['InfectedPopulation_Ratio'] = train_data_by_country_confirm['InfectedPopulation_Ratio'].replace([-np.inf, np.inf],  0.0)


train_data_by_country_confirm.set_index('Country_Region', inplace=True)
train_data_by_country_confirm.style.background_gradient(cmap='Reds').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}", 'InfectedPopulation_Ratio': "{:.4f}"})



train_data_by_country_confirm['DeceasedPopulation_Ratio'] = train_data_by_country_confirm['Fatalities'].div(train_data_by_country_confirm['Population'],fill_value=0)*100
train_data_by_country_confirm['DeceasedPopulation_Ratio'] = train_data_by_country_confirm['DeceasedPopulation_Ratio'].replace([-np.inf, np.inf],  0.0)


train_data_by_country_confirm= train_data_by_country_confirm.sort_values('DeceasedPopulation_Ratio', ascending=False).reset_index()
train_data_by_country_confirm.style.background_gradient(cmap='Reds').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}", 'InfectedPopulation_Ratio': "{:.4f}", 'DeceasedPopulation_Ratio':"{:.4f}"})


train_data_by_country_confirm= train_data_by_country_confirm.sort_values('GrowthRate', ascending=False)
train_data_by_country_confirm.style.background_gradient(cmap='Reds').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}", 'InfectedPopulation_Ratio': "{:.4f}"})

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
import folium 
from folium import plugins
from tqdm.notebook import tqdm as tqdm

train_data_by_country_confirm= train_data_by_country_confirm.reset_index()
global_confirmedcases = train_data_by_country_confirm[['Country_Region','InfectedPopulation_Ratio']]

fig = px.bar(global_confirmedcases.sort_values('InfectedPopulation_Ratio',ascending=False)[:20],x='InfectedPopulation_Ratio',y='Country_Region',title='Country wise infected population ratio',text='InfectedPopulation_Ratio', height=900, orientation='h')
fig.show()

# without diamond princess
rest = global_confirmedcases[global_confirmedcases['Country_Region'] != 'Diamond Princess'].sort_values('InfectedPopulation_Ratio',ascending=False)[1:20]
fig = px.bar(rest,x='InfectedPopulation_Ratio',y='Country_Region',title='Country wise infected population ratio excluding Diamond Princess',text='InfectedPopulation_Ratio', height=900, orientation='h')
fig.show()
cleaned_data_mitigation = pd.read_csv('../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv', parse_dates=['Date Start','Date end intended' ])
total_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = confirmed - deaths - recovered
total_data['Active'] = total_data['Confirmed'] - total_data['Deaths'] - total_data['Recovered']

# filling missing values 
total_data[['Province/State']] = total_data[['Province/State']].fillna('')
total_data[cases] = total_data[cases].fillna(0)
total_data.head()
cleaned_data_mitigation = cleaned_data_mitigation.rename(columns={"Date Start": "Date", 'Country':'Country/Region'})
# row = cleaned_data_mitigation[(cleaned_data_mitigation['Country/Region']=='Iran') & (cleaned_data_mitigation['Date']=='2020-03-17')]
# str = pd.DataFrame(pd.Series(['shutdown']))
# print(type(row['Keywords']))
# row['Keywords'] = row['Keywords'].append(str, )
# print(row['Keywords'])
# print(cleaned_data_mitigation[(cleaned_data_mitigation['Country/Region']=='Iran') & (cleaned_data_mitigation['Date']=='2020-03-17')])
# cleaned_data_mitigation = pd.concat([row, cleaned_data_mitigation], ignore_index=True)
# including the mitigation measure for Iran


cont_spread = cleaned_data_mitigation.merge(total_data, how='outer', on=["Date", 'Country/Region'] )

print(len(cont_spread))
# # cond2  = cont_spread.Fatalities.notnull()
# cond2 = cont_spread['Date']=='2020-03-09'
ttc = cont_spread.loc[(cont_spread['Country/Region'] == 'South Korea')]
display(ttc)
countries = ['South Korea','Germany','Iceland','Iran','Italy', 'Spain', 'Russia',  'Japan', 'China', 'United Kingdom', 'United States','Netherlands', 'France']


def country_cont(data, country):
    df = data[data['Country/Region'] == country][['Date',
                                    'Description of measure implemented',
                                    'Exceptions', 
                                    'Keywords', 
                                    'Target region']].copy()
    df['region'] = df['Target region'].fillna('All')
    df['Keywords'] = df['Keywords'].fillna('-')
#     df['date'] = pd.to_datetime(df['Date'])
#     df.drop(['Date', 'Target region'], axis=1, inplace=True)
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    return df

cont_all = []

for country in countries:
    tmp = country_cont(cont_spread, country)
    tmp['Country/Region'] = country
    tmp = tmp[tmp.Keywords.str.contains('lockdown|business suspension|school closure|travel ban|social distancing|blanket curfew|shutdown|emergency|nonessential|outdoor gatherings banned|contact tracing|traveller screening|remote work|personal hygiene')]
#     tmp= tmp[tmp.Keywords.str.contains('\a*', regex= True)]
    cont_all.append(tmp[['Country/Region', 'Date', 'Keywords', 'region']])
    
cont_all = pd.concat(cont_all, ignore_index=False)

cont_spread2 = cont_all.merge(total_data, how='outer', on=["Date", "Country/Region"] )

cond2 = cont_spread2['Date']=='2020-03-22'
cont_spreadwq = cont_spread2.loc[(cont_spread2['Country/Region']=='South Korea') & (cond2)]
cont_spreadwq.head()

# without the keyword labels
for country in countries:
 
 grouped_country = cont_spread2[cont_spread2['Country/Region'] == country].reset_index()
 grouped_country_date = grouped_country.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'
                                                  ].sum().reset_index()

 fig = make_subplots(rows=1, cols=1)
 trace4 = go.Scatter(x=grouped_country_date['Date'],y=grouped_country_date['Active'],name="Active",
                    line_color='green',mode='lines+markers',opacity=0.8)
 fig.append_trace(trace4, 1, 1)

 fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in {}</b>'.format(country),
                   height= 400,font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
#  fig.show()
for country in countries:
 
 grouped_country = cont_spread2[cont_spread2['Country/Region'] == country].reset_index()
 grouped_country_date = grouped_country
#  grouped_country_date = grouped_country.groupby('Date')['Date', 'Confirmed', 'Deaths','Recovered','Active'
#                                                   ].sum().reset_index()
#  grouped_country_date['Keywords']= grouped_country['Keywords']

# keywords = []
# for date in grouped_country_date['Date']:
#     keyword_value = ''
#     for dateStart in grouped_country['Date']:
#         if date == dateStart:
#             keyword_value = grouped_country[grouped_country['Date'] == dateStart]['Keywords']
#             break
#     keywords.append(keyword_value)
# grouped_country_date['Keywords'] = keywords
# print(keywords)

 fig = make_subplots(rows=1, cols=1)
#  trace1 = go.Scatter(x=grouped_country_date['Date'],y=grouped_country_date['Confirmed'],name="Confirmed",hovertext=grouped_country_date['Keywords']
#                     ,line_color='yellow',mode='lines+markers',opacity=0.8)
#  trace2 = go.Scatter(x=grouped_country_date['Date'],y=grouped_country_date['Deaths'],name="Deaths",hovertext=grouped_country_date['Keywords'],
#                     line_color='orange',mode='lines+markers',opacity=0.8)
#  trace3 = go.Scatter(x=grouped_country_date['Date'],y=grouped_country_date['Recovered'],name="Recovered",
#                      line_color='red',mode='lines+markers',opacity=0.8)
 trace4 = go.Scatter(x=grouped_country_date['Date'],y=grouped_country_date['Active'],name="Active",hovertext=grouped_country_date['Keywords'],
                    line_color='green',mode='lines+markers',opacity=0.8)
#  fig.append_trace(trace1, 1, 1)
#  fig.append_trace(trace2, 2, 1)
#  fig.append_trace(trace3, 3, 1)
 fig.append_trace(trace4, 1, 1)

 fig.update_layout(template="plotly_dark",title_text = '<b>Spread of the COVID19 over time in {}</b>'.format(country),
                   height= 400,font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
 fig.show()
import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit

from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def min_date(data, num, text, by=None):
    
    if by is None:
        by = 'Country_Region'
    
    min_dates = data[data['n_cases'] > num].groupby(by, as_index=False).date.min()
    
    return min_dates.rename(columns={'date': text})


def increase(data):
    increase = data[['Country_Region', 'date', 'n_cases']].sort_values(['Country_Region', 'date'], ascending=True).copy()
    increase['increase'] = increase.n_cases.diff().fillna(0)
    increase.loc[increase.increase < 0, 'increase'] = 0
    increase['perc_increase'] = (increase.increase / (increase.n_cases - increase.increase) * 100).fillna(0)
    
    return increase[['Country_Region', 'date', 'increase', 'perc_increase']]


def load_series(path):
    ignore = ['Province/State', 'Lat', 'Long']
    data = pd.read_csv(path)
    
    data = data[[col for col in data if col not in ignore]].groupby('Country/Region', as_index=False).sum()
    data.rename(columns={'Country/Region': 'Country_Region'}, inplace=True)
    
    data = data.melt(id_vars='Country_Region')
    data['date'] = pd.to_datetime(data['variable'])
    del data['variable']
    data.rename(columns={'value': 'n_cases'}, inplace=True)
    
    first_case = min_date(data, 0, 'first_date')
    data = pd.merge(data, first_case, on='Country_Region', how='left')
    data['from_first'] = (data['date'] - data['first_date']).dt.days
    
    case_10 = min_date(data, 9, '10th_date')
    data = pd.merge(data, case_10, on='Country_Region', how='left')
    data['from_10th'] = (data['date'] - data['10th_date']).dt.days
    
    case_50 = min_date(data, 49, '50th_date')
    data = pd.merge(data, case_50, on='Country_Region', how='left')
    data['from_50th'] = (data['date'] - data['50th_date']).dt.days
    
    case_100 = min_date(data, 99, '100th_date')
    data = pd.merge(data, case_100, on='Country_Region', how='left')
    data['from_100th'] = (data['date'] - data['100th_date']).dt.days
    
    case_500 = min_date(data, 499, '500th_date')
    data = pd.merge(data, case_500, on='Country_Region', how='left')
    data['from_500th'] = (data['date'] - data['500th_date']).dt.days
    
    data['Country_Region'] = data['Country_Region'].map({'US': 'United States', 
                                                         'Korea, South': 'South Korea'}).fillna(data['Country_Region'])
    
    continents = pd.read_csv('/kaggle/input/country-to-continent/countryContinent.csv', encoding = 'ISO-8859-1')
    continents['country'] = continents['country'].map({'United States of America': 'United States', 
                                                       'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
                                                       'Korea (Republic of)': 'South Korea', 
                                                       "Korea (Democratic People's Republic of)": 'North Korea'}).fillna(continents['country'])
    
    data = pd.merge(data, continents[['country', 'continent', 'sub_region']], left_on='Country_Region', right_on='country', how='left')
    del data['country']
    
    country_info = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
    data = pd.merge(data, country_info[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Med. Age']], 
                    left_on='Country_Region', right_on='Country (or dependency)', how='left')
    data.rename(columns={'Population (2020)': 'population', 'Density (P/Km²)': 'pop_density', 'Med. Age': 'median_age'}, inplace=True)
    data.loc[data.median_age == 'N.A.', 'median_age'] = np.nan
    data['median_age'] = pd.to_numeric(data.median_age)
    del data['Country (or dependency)']
    
    new_cases = increase(data)
    data = pd.merge(data, new_cases, on=['Country_Region', 'date'], how='left')
    
    return data


def start_from(cases, deaths, col_start, on_cases=False, n_top=10, true_count='x', make_rate=False, ch_date=True):
    if on_cases:
        tmp = pd.merge(cases[['Country_Region', 'date', 'continent', 'population', 'n_cases']+[col_start]], 
                      deaths[['Country_Region', 'date', 'n_cases']], on=['Country_Region', 'date'])
    else:
        tmp = pd.merge(cases[['Country_Region', 'date', 'continent', 'population', 'n_cases']], 
                      deaths[['Country_Region', 'date', 'n_cases']+[col_start]], on=['Country_Region', 'date'])
        
    top_countries = tmp.groupby('Country_Region').n_cases_x.max().sort_values(ascending=True).tail(n_top).index.tolist()
    tmp = tmp[tmp[col_start] > 0]
    tmp = tmp[tmp.Country_Region.isin(top_countries)]
    if ch_date:
        tmp['date'] = tmp[col_start]
    if make_rate:
        tmp['n_cases_y'] = (tmp['n_cases_y'] / tmp['n_cases_x'] * 100).fillna(0)
        true_count = 'y'
    
    tmp['n_cases'] = tmp[f'n_cases_{true_count}']
    
    return tmp


def country_cont(data, country):
    df = data[data.Country == country][['Date Start',
                                    'Description of measure implemented',
                                    'Exceptions', 
                                    'Keywords', 
                                    'Target region']].copy()
    df['region'] = df['Target region'].fillna('All')
    df['Keywords'] = df['Keywords'].fillna('-')
    df['date'] = pd.to_datetime(df['Date Start'])
    df.drop(['Date Start', 'Target region'], axis=1, inplace=True)
    df = df.sort_values(by='date').reset_index(drop=True)
    
    return df
conf_cases = load_series('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
recovered = load_series('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
deaths = load_series('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

# containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID%2019%20Containment%20measures%202020-03-30.csv')
# tmp = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID-19%20containment%20measures/COVID 19 Containment measures data.csv')
# containment = pd.concat([containment, tmp], ignore_index=True)
# containment.loc[containment.Country.fillna('-').str.contains('US:'), 'Country'] = 'United States'
containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')
containment.loc[containment.Country.fillna('-').str.contains('US:'), 'Country'] = 'United States'

conf_cases[(conf_cases.median_age <45) &(conf_cases.median_age >20)]
to_use = ['Country_Region', 'n_cases', 'increase', 'perc_increase', 'date', 'first_date', 'from_first', 
          '10th_date', 'from_10th', '50th_date', 'from_50th', '100th_date', 'from_100th', '500th_date', 'from_500th']

full_data = pd.merge(conf_cases[to_use].rename(columns={'first_date': 'first_case_date', 
                                                        'from_first': 'from_first_case', 
                                                        '10th_date': '10th_case_date', 
                                                        'from_10th': 'from_10th_case', 
                                                        '50th_date': '10th_case_date', 
                                                        'from_50th': 'from_50th_date',
                                                        '100th_date': '100th_case_date', 
                                                        'from_100th': 'from_100th_case', 
                                                        '500th_date': '500th_case_date',
                                                        'from_500th': 'from_500th_case', 
                                                        'increase': 'new_cases', 
                                                        'perc_increase': 'new_cases_perc'}), 
                     deaths[to_use].rename(columns={'first_date': 'first_victim_date', 
                                                        'from_first': 'from_first_victim', 
                                                        '10th_date': '10th_victim_date', 
                                                        'from_10th': 'from_10th_victim', 
                                                        '50th_date': '50th_victim_date', 
                                                        'from_50th': 'from_50th_victim',
                                                        '100th_date': '100th_victim_date', 
                                                        'from_100th': 'from_100th_victim', 
                                                        '500th_date': '500th_victim_date',
                                                        'from_500th': 'from_500th_victim', 
                                                    'n_cases': 'n_victims', 
                                                        'increase': 'new_victims', 
                                                        'perc_increase': 'new_victims_perc'}), 
                    on=['Country_Region', 'date'])


full_data.head()
measures ='lockdown|business suspension|school closure|travel ban|social distancing|blanket curfew|shutdown|emergency|nonessential|outdoor gatherings banned|contact tracing|traveller screening|remote work|personal hygiene'
countries = ['South Korea','Germany', 'Iceland','Iran','Italy', 'Spain', 'Russia',  'Japan', 'United Kingdom','Netherlands', 'France']


cont_all = []

for country in countries:
    tmp = country_cont(containment, country)
    tmp['Country_Region'] = country
    tmp['date'] = tmp['date'].fillna('-')
    tmp = tmp[tmp.Keywords.str.contains(measures)]
    cont_all.append(tmp[['Country_Region', 'date', 'Keywords', 'region']])
    
cont_all = pd.concat(cont_all, ignore_index=False)
# cont_all = cont_all[(cont_all.date >= pd.to_datetime('2020-02-01'))]  # a school ban that I can't find proof of

cont_all.head()
from scipy.signal import argrelextrema
def pop(df):
    Population =[]
    Density= []
    countries = df['Country_Region'].values
    
    countries = list(countries)
    for country in countries:
#       updata.groupby('Country_Region')[Population]
#         print(updata[updata['Country_Region']==country])
        pop = updata[updata['Country_Region']==country].Population.values
        density =updata[updata['Country_Region']==country].density.values
        Population.append(pop)
        Density.append(density)
    df['Population'] =  Population   
    df['Density'] = Density
    return df
def getDaysTaken(df):
    measure_Date = []
    peak_Date = []
    Peak_cases = []
    Days =[] 
    countries = df['Country_Region'].values
    
    countries = list(countries)
    for country in countries:
     grouped_country = cont_spread2[cont_spread2['Country/Region'] == country].reset_index()
     grouped_country_date = grouped_country
     n=6 # number of points to be checked before and after 
# Find local peaks
     grouped_country_date['min'] = grouped_country_date.iloc[argrelextrema(grouped_country_date.Active.values, np.less_equal, order=n)[0]]['Active']
     grouped_country_date['max'] = grouped_country_date.iloc[argrelextrema(grouped_country_date.Active.values, np.greater_equal, order=n)[0]]['Active']
     maxpeak= grouped_country_date['max'].max()
#      print(maxpeak)
     pdate = grouped_country_date[grouped_country_date.Active == maxpeak].Date
     
#      print(country)
     
     date_first= res[res['Country_Region']==country].date.values
#      print('measure date:{}'.format(pd.Timestamp(date_first[0])))

     date_peak = pdate.values
#      print('peak date:{}'.format(pd.Timestamp(date_peak[0])))
     
     days = (pd.Timestamp(date_peak[0]) - pd.Timestamp(date_first[0]))
#      print(days)
     
#      print()
     measure_Date.append(pd.Timestamp(date_first[0]))
     peak_Date.append(pd.Timestamp(date_peak[0]))
     Days.append(days)
     Peak_cases.append(maxpeak)
    df['measure_Date'] = measure_Date
    df['peak_Date'] =peak_Date
    df['Days']= Days
    df['Peak_cases'] =Peak_cases
      
    return res

low_measures = 'school closure|gathering|travel ban|hygiene'
med_measures = 'social distancing|remote work'
crit_measures = 'blanket|lockdown|emergency|stay|tracing|isolation'

tmp = cont_all[cont_all.Keywords.str.contains(low_measures)].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))


med_measures = 'social distancing|remote work'
tmp = cont_all[cont_all.Keywords.str.contains(med_measures)].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))
crit_measures = 'blanket|lockdown|emergency|stay|tracing|isolation'

tmp = cont_all[cont_all.Keywords.str.contains(crit_measures)].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))



tmp = cont_all[cont_all.Keywords.str.contains('travel|hygiene')].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))



tmp = cont_all[cont_all.Keywords.str.contains('social distancing')].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))


tmp = cont_all[cont_all.Keywords.str.contains('isolation|blanket')].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))
tmp = cont_all[cont_all.Keywords.str.contains('lockdown|curfew|isolation')].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
crit =tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
getDaysTaken(crit)
pop(crit)
tmp = cont_all[cont_all.Keywords.str.contains('social distancing|isolation|blanket|hygiene')].groupby('Country_Region', as_index=False).date.min()

tmp = pd.merge(tmp, full_data, on=['Country_Region', 'date'])
res= tmp[['Country_Region', 'date','n_cases', 'n_victims', 'from_10th_case', 'from_first_victim', 'from_100th_case', 'from_10th_victim']].sort_values('n_cases')
display(getDaysTaken(res))
display(pop(res))
train_data_by_country = train_data.groupby(['Date','Country_Region'],as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum',
                                                                                         'GrowthRate':'last' })
#display(train_data_by_country.tail(10))
max_train_date = train_data['Date'].max()
train_data_by_country_confirm = train_data_by_country.query('(Date == @max_train_date) & (ConfirmedCases > 100)').sort_values('ConfirmedCases', ascending=False)
train_data_by_country_confirm.set_index('Country_Region', inplace=True)

train_data_by_country_confirm.style.background_gradient(cmap='Reds').format({'ConfirmedCases': "{:.0f}", 'GrowthRate': "{:.2f}"})




