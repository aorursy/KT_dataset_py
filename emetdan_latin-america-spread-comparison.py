# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# importing datasets

full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)
# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
# Latinoamerica

latam_f = temp_f[temp_f['Country/Region'].isin(['Colombia', 'Brazil', 'Chile', 'Argentina',

                                             'Panama',  'Peru', 'Mexico',

                                             'Ecuador', 'Costa Rica', 'Venezuela',

                                              'Bolivia', 'Paraguay', 'Uruguay',

                                             'Cuba', 'Guyana', 'Honduras',

                                             'Dominican Republic', 'Jamaica', 'Guatemala',

                                             'El Salvador'])]

latam_f.style.background_gradient(cmap='Reds')
# Exponetial curves

exp_22 = (27.0/100 + 1)**np.arange(0, 50, 1)

p = 22.0

days = np.arange(0, 21, 0.5)

exp = (p/100 + 1)**days - 1
temp = full_table.groupby(['Date', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)



# Colombia, Perú  -> 7 marzo

# Brazil -> 28 febrero

# Panama -> 10 marzo

# Ecuador -> 2 marzo

# Argentina -> 4 marzo

# Chile -> 3 marzo

# Mexico -> 28 febrero



# relative country cases

def rel_country_cases(country, zero_case_date, population):  

    c = temp[temp['Country/Region']==country]

    c.rename(columns={'Country/Region':'País'}, inplace=True)

    c = c.sort_values('Date', ascending=True)

    c = c.set_index(['Date'])

    c = c.loc[zero_case_date:'2020-3-25']

    print(country, len(c))

    c = c.reset_index()

    c['Casos por Millón de Habitantes'] = c['Confirmed']/(population)

    c['Días Después del Caso 10'] = range(len(c))

    

    return c



col = rel_country_cases('Colombia', '2020-3-13', 49) # 2020-3-6, 49 mill

per = rel_country_cases('Peru', '2020-3-10', 32) # 2020-3-6, 32 mill

bra = rel_country_cases('Brazil', '2020-3-6', 209) # 2020-2-26, 209 mill

pan = rel_country_cases('Panama', '2020-3-12', 4) # , 4 mill

ec = rel_country_cases('Ecuador', '2020-3-5', 16.6) # , 16.6 mill

ar = rel_country_cases('Argentina', '2020-3-8', 44.2) # 2020-3-4, 44.2 mill

chi = rel_country_cases('Chile', '2020-3-10', 18.05) # , 18.05 mill

mex = rel_country_cases('Mexico', '2020-3-12', 129) # , 129 mill

it = rel_country_cases('Italy', '2020-2-21', 60.4) # 2020-1-31, 2020-2-18, 60.4 mill

us = rel_country_cases('US', '2020-2-24', 327) # 327 mill

spa = rel_country_cases('Spain', '2020-2-26', 46.6) # 46.6 mill

exp_df = pd.DataFrame(data={'Días Después del Caso 10':days, 'Casos por Millón de Habitantes':exp, 'País':f'{int(p)}% Incremento Diario'})



#latam = pd.concat([col, per, bra, pan, ec, ar, chi, mex]) 

latam = pd.concat([col, ec, per, ar, chi, pan, mex, bra, spa, it])

fig = px.line(latam, x='Días Después del Caso 10', y='Casos por Millón de Habitantes',

              color='País', title='Diseminación de Casos COVID-19', height=600, width=1200, template="none")

fig.update_traces(mode='lines+markers')

fig.update_layout(yaxis_type="log")
temp[temp['Country/Region'] == 'Ecuador'].sort_values('Date', ascending=False).head(50)