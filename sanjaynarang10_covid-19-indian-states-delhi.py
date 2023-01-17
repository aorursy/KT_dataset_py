import numpy

import pandas as pd

from pandas import DataFrame

from pandas.io.json import json_normalize

import calendar

import os

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt
# To plot pretty figures

%matplotlib inline
# load dataset

raw_data = pd.read_json("https://api.rootnet.in/covid19-in/stats/history")

raw_data.shape
my_state = ["Delhi"] # The state for which analysis is to be conducted.
framed_data = DataFrame(raw_data)

framed_data = framed_data['data']
original_data = json_normalize(framed_data,'regional',['day'])
original_data = original_data.rename(columns={"day": "date_report"})

original_data["date_report"] = pd.to_datetime(original_data["date_report"])

original_data.shape

original_data.info()
#original_data = original_data[(original_data['confirmedCasesIndian'] > 2000)]
my_state_data = original_data[(original_data['loc'].isin(my_state))]

my_state_data.info()
original_data.set_index('date_report')
my_state_data['total_active_cases'] = my_state_data['totalConfirmed'] - my_state_data['deaths'] - my_state_data['discharged']
my_state_data['daily_new_cases'] = my_state_data['confirmedCasesIndian'] - my_state_data['confirmedCasesIndian'].shift()
my_state_data.describe()
g1_states_data = my_state_data[['date_report','totalConfirmed','deaths','discharged','total_active_cases']]
g1_states_data.head(100)
g1_states_data.set_index(['date_report'], inplace=True)
import matplotlib.ticker as ticker



colors = {'totalConfirmed':'#045275', 'deaths':'#089099', 'discharged':'#7CCBA2', 'total_active_cases':'#7C1D6F'}



plt.style.use('fivethirtyeight')



plot = g1_states_data.plot(figsize=(8,8), color=list(colors.values()), linewidth=5, legend=False, kind="line")



plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))



plot.grid(color='#d4d4d4')



plot.set_xlabel('report date')



plot.set_ylabel('# of cases')

plot.set_title('Covid situation in '+ my_state[0])



# Section 8 - Assigning Colour



for case_type in list(colors.keys()):



    plot.text(x = g1_states_data.index[-1], y = g1_states_data[case_type].max(), color = colors[case_type], s = case_type, weight = 'bold')

    
g2_states_data = my_state_data[['date_report','daily_new_cases']]

g2_states_data = g2_states_data[g2_states_data['daily_new_cases']>= 200]
g2_states_data.set_index(['date_report'], inplace=True)

#g2_states_data.head(100)
#x = DataFrame(g2_states_data.groupby([g2_states_data['date_report'].dt.strftime('%W')])['daily_new_cases'].sum())

sns.set_style("whitegrid")

plt.style.use('fivethirtyeight')

plot = g2_states_data.plot.bar(rot="45",figsize=(20,20))



plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))



plot.grid(color='#089099')



plot.set_xlabel('date report')



plot.set_ylabel('# of cases')

plot.set_title('Daily new cases in ' + my_state[0])
fig = plt.figure(figsize=(12,8))



plot = sns.kdeplot(my_state_data['total_active_cases'],my_state_data['discharged'])

plot.set_xlabel('Total Active Cases')



plot.set_ylabel('# of Discharged')

plot.set_title('Total Active & Discharged in ' + my_state[0])
fig = plt.figure(figsize=(12,8))



plot = sns.kdeplot(my_state_data['confirmedCasesIndian'],my_state_data['deaths'])

plot.set_xlabel('Total Confirmed Cases')



plot.set_ylabel('# of Deaths')

plot.set_title('Total Confirmed Cases & Deaths in ' + my_state[0])
mobility_data = pd.read_csv("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv")

countries = ["India"]

i_m_d = mobility_data[mobility_data['country_region'].isin(countries)]

i_m_d = i_m_d.drop(columns=['country_region_code','country_region','sub_region_2'])
top_states = ["Delhi","Maharashtra","Tamil Nadu","Gujarat","Kerala"]



i_m_d = i_m_d[i_m_d['sub_region_1'].isin(top_states)]

i_m_d['date_report'] = pd.to_datetime(i_m_d['date'])

i_m_d.drop(columns= ['date'])
i_m_d_states = i_m_d.copy()
i_m_d_states = i_m_d_states.drop(columns=['iso_3166_2_code','census_fips_code','date'])

i_m_d_states = i_m_d_states[i_m_d_states['sub_region_1'].isin(my_state)]
i_m_d_states.set_index(['date_report'], inplace=True)
plot = i_m_d_states.plot(figsize=(18,12), linewidth=5, legend=True)

plot.set_xlabel('report date')



plot.set_ylabel('Change in mobility')

plot.set_title('Change in mobilities since lockdwon in ' + my_state[0])
g5_states_data = original_data[original_data['loc'].isin(top_states)]

cases_mobility_data = pd.merge(g5_states_data,i_m_d,how="right",left_on=['loc','date_report'],right_on=['sub_region_1','date_report'])
cases_mobility_data["average_mobility"] = cases_mobility_data[['retail_and_recreation_percent_change_from_baseline','grocery_and_pharmacy_percent_change_from_baseline','parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline','workplaces_percent_change_from_baseline','residential_percent_change_from_baseline']].mean(axis=1)
cases_mobility_data = cases_mobility_data[['date_report','sub_region_1','average_mobility']]
cases_mobility_data = cases_mobility_data.pivot(index='date_report',columns='sub_region_1', values='average_mobility')
state_columns = list(cases_mobility_data.columns)
cases_mobility_data = cases_mobility_data.reset_index('date_report')
cases_mobility_data.set_index(['date_report'], inplace=True)
colors = {'Delhi':'#045275', 'Gujarat':'#089099', 'Kerala':'#7CCBA2', 'Maharashtra':'#7C1D6F','Tamil Nadu':'#DC3977'}



plot = cases_mobility_data.plot(figsize=(18,12), color=list(colors.values()), linewidth=5, legend=True)



plot.set_xlabel('report date')



plot.set_ylabel('change in mobility')

plot.set_title('Change in average mobility from basline in top 5 states')

g6_states_data = original_data[original_data['loc'].isin(state_columns)]
g6_states_data = g6_states_data.pivot(index='date_report', columns='loc', values='confirmedCasesIndian')
covid = g6_states_data.reset_index('date_report')
covid.set_index(['date_report'], inplace=True)
covid.columns = state_columns
import matplotlib.ticker as ticker



colors = {'Delhi':'#045275', 'Maharashtra':'#089099', 'Tamil Nadu':'#7CCBA2', 'Gujarat':'#7C1D6F', 'Kerala':'#DC3977'}



plt.style.use('fivethirtyeight')



plot = covid.plot(figsize=(12,8), color=list(colors.values()), linewidth=5, legend=False, kind="line")



plot.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))



plot.grid(color='#d4d4d4')



plot.set_xlabel('report date')



plot.set_ylabel('# of new Cases')

plot.set_title('New Cases in Top 5 States')





for state in list(colors.keys()):



    plot.text(x = covid.index[-1], y = covid[state].max(), color = colors[state], s = state, weight = 'bold')


