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
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

%matplotlib inline
# source: https://www.kaggle.com/eyalgal/dashboarding-covid-the-good-the-bad-the-ugly

# filename_pattern = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{}.csv'
filename_pattern = 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{}.csv'
filename_pattern = "../input/novel-corona-virus-2019-dataset/time_series_covid_19_{}.csv"

# time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
# time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
# time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")


confirmed = pd.read_csv(filename_pattern.format('confirmed')).set_index(['Province/State','Country/Region', 'Lat', 'Long'])
recovered = pd.read_csv(filename_pattern.format('recovered')).set_index(['Province/State','Country/Region', 'Lat', 'Long'])
deaths = pd.read_csv(filename_pattern.format('deaths')).set_index(['Province/State','Country/Region', 'Lat', 'Long'])

print(f'Dates: From {confirmed.columns[0]} to {confirmed.columns[-1]}.')
# confirmed

covid = pd.concat([pd.read_csv(filename_pattern.format(name.lower())) \
                       .melt(id_vars=['Province/State','Country/Region', 'Lat', 'Long'], var_name='Date', value_name=name) \
                       .set_index(['Province/State','Country/Region', 'Lat', 'Long', 'Date'])
                   for name in ['Confirmed', 'Recovered', 'Deaths']], axis=1).reset_index() \
    .assign(Infected=lambda df: df['Confirmed']-df['Recovered']-df['Deaths'])
covid['Date'] = pd.to_datetime(covid['Date'])
covid
covid_world = covid.groupby('Date')[['Confirmed', 'Recovered', 'Deaths', 'Infected']].sum()
#covid_world.iloc[[-1]].style.format('{:,}')
covid_world

df = covid_world.reset_index().melt(id_vars='Date', var_name='Status', value_name='Subjects')
px.line(df, 'Date', 'Subjects', color='Status', title='World wide trends')
import re
covid_countries = covid.groupby(['Country/Region', 'Date'])['Confirmed', 'Recovered', 'Deaths', 'Infected'].sum()
# covid_countries
countries = [c for c in covid['Country/Region'].unique() 
#             if re.search(r'China|Korea|Italy|Iran|US|Argentina|Germany', c)]
             if re.search(r'Korea|Italy|US|Argentina|Germany', c)]

df = covid_countries.reset_index().melt(id_vars=['Date', 'Country/Region'], var_name='Status', value_name='Subjects')

fig = px.line(df[df['Country/Region'].isin(countries)], 'Date', 'Subjects', 
              color='Status', facet_col='Country/Region', 
              facet_col_wrap=2, height=600)
fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_layout(legend_orientation="h")
country = confirmed.groupby(['Country/Region']).sum()
country
latest_confirmed = country[country.columns[-1]].sort_values(ascending=False).to_frame('Confirmed').reset_index()
display(latest_confirmed)
px.bar(latest_confirmed.nlargest(200, 'Confirmed'), x='Country/Region', y='Confirmed')
fig = px.scatter(df[(df['Country/Region'].isin(latest_confirmed.nlargest(20, 'Confirmed')['Country/Region'][1:])) &
                 (df['Status']=='Confirmed')], 
                 'Date', 'Subjects', color='Country/Region', 
                 log_y=True, height=600)
fig.update_traces(mode='lines+markers', line=dict(width=.5))
# fig.update_layout(legend_orientation="h")


fig = px.scatter(df[(df['Country/Region'].isin(countries)) &
                 (df['Status']=='Confirmed')], 
                 'Date', 'Subjects', color='Country/Region', 
                 log_y=True, height=600)
fig.update_traces(mode='lines+markers', line=dict(width=.5))
# fig.update_layout(legend_orientation="h")
# fig.update_traces(mode='lines+markers', line=dict(width=.5))
# # fig.update_layout(legend_orientation="h")

countries = [c for c in covid['Country/Region'].unique() 
             if re.search(r'Argentina|Italy|Spain|US|Germany|Brazil|Uruguay|Chile', c)]

df_cases = covid_countries.reset_index().melt(id_vars=['Date', 'Country/Region'], var_name='Status', value_name='Subjects')
df_cases = df_cases[df_cases['Country/Region'].isin(countries)]

#TODO: Fix population source
population_date = "06Mar2020"
_dict = {
    "Global": "7 738 323 220",
    "China": "1 405 371 596",
    "Japan": "125 406 227",
    "South Korea": "51 277 160",
    "Italy": "59 813 196",
    "Iran": "83 473 631",
    "Argentina": "45 436 302",
    "Spain": "45 852 322",
    "Germany": "80 659 320",
    "US": "332 838 247",
    "Brazil": "215 964 604",
    "Uruguay": "3 490 607",
    "Chile": "18 814 916",
}
population_dict = {k: int(v.replace(" ", "")) for (k, v) in _dict.items()}
df_pop = pd.io.json.json_normalize(population_dict)
df_pop.index = [f"Total population on {population_date}"]
# df_cases_per_habitant = df_cases['Subjects']/df_pop
df_by_country=[df_cases[df_cases['Country/Region'].isin([c])] for c in countries]
df_with_cases_per_100000_list = []
for df in df_by_country:
    numerator = df['Subjects']
    denominator = pd.Series.repeat(df_pop[df['Country/Region'].iloc[0]], len(df.index))
    df.insert(4, 'Population', pd.DataFrame(denominator).to_numpy())
    df.insert(5, 'Cases per 100000', df['Subjects']/df['Population']*100000)
    df_with_cases_per_100000_list.append(df)
    
#     print('denominator')
#     print(denominator)
#     print('numerator')
#     print(numerator)
#     print('-------------')
#     print('.')
#    print(numerator.div(denominator, axis='columns'))
#    temp_df
df_with_cases_per_100000 = pd.concat(df_with_cases_per_100000_list)
df_with_cases_per_100000
fig = px.line(df_with_cases_per_100000[df_with_cases_per_100000['Status']=='Confirmed'], 'Date', 'Cases per 100000', 
              color='Status', facet_col='Country/Region', 
              facet_col_wrap=2, height=600)
fig.update_yaxes(matches=None, showticklabels=True)
fig = px.scatter(df_with_cases_per_100000[df_with_cases_per_100000['Status']=='Confirmed'], 
                 'Date', 'Cases per 100000', color='Country/Region', 
                 log_y=True, height=600)
fig.update_traces(mode='lines+markers', line=dict(width=.5))
fig = px.line(df_with_cases_per_100000[df_with_cases_per_100000['Status']=='Deaths'], 'Date', 'Cases per 100000', 
              color='Status', facet_col='Country/Region', 
              facet_col_wrap=2, height=600)
fig.update_yaxes(matches=None, showticklabels=True)
fig = px.scatter(df_with_cases_per_100000[df_with_cases_per_100000['Status']=='Deaths'], 
                 'Date', 'Cases per 100000', color='Country/Region', 
                 log_y=True, height=600, title='Deaths per 100000:')
fig.update_traces(mode='lines+markers', line=dict(width=.5))

