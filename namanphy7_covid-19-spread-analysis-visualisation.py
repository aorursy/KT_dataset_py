import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white", color_codes=True)



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# We will use - 'corona-virus-report', 'novel-corona-virus-2019-dataset' 
# import os

# print(os.listdir("../input/"))

# for item in os.listdir("../input/"):

#     print(os.listdir(f'../input/{item}'))
table1 = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

table1 = table1.dropna(axis=0, how='all')

print(table1.shape)

table1.head(2)
table2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')



# Removing columns that might be not necessary

table2 = table2.drop(['Unnamed: 33','Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37','Unnamed: 38', 

                      'Unnamed: 39','Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44',

                      'location', 'admin3', 'admin2', 'admin1','country_new', 'admin_id', 'data_moderator_initials'], axis=1)

table2 = table2.dropna(axis=0, how='all')

print(table2.shape)

table2.head(2)
#PREPRARING TABLE1 DATA



cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case: Active Case = confirmed - deaths - recovered

table1['Active'] = table1['Confirmed'] - table1['Deaths'] - table1['Recovered']



# Renaming Mainland china as China in the data table

table1['Country/Region'] = table1['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

table1[['Province/State']] = table1[['Province/State']].fillna('')

table1[cases] = table1[cases].fillna(0)



# latest

table1_latest = table1[table1['Date'] == max(table1['Date'])].reset_index()

# latest_reduced

table1_latest_gp = table1_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

table1.head(2)
# PREPARING TABLE2 DATA



# Filling null values with a past date (to detect it easily) and removing irregular datetime values

table2_time = table2[['country','date_onset_symptoms', 'date_admission_hospital', 'date_confirmation']].fillna('01.01.2000')

table2_time = table2_time.drop(table2_time[table2_time['date_onset_symptoms'].str.len() != 10].index)

table2_time = table2_time.drop(table2_time[table2_time['date_admission_hospital'].str.len() != 10].index)

table2_time = table2_time.drop(table2_time[table2_time['date_confirmation'].str.len() != 10].index)



# converting to datetime

table2_time['date_onset_symptoms'] = pd.to_datetime(table2_time['date_onset_symptoms'].str.strip(), format='%d.%m.%Y', errors='coerce')

table2_time['date_admission_hospital'] = pd.to_datetime(table2_time['date_admission_hospital'].str.strip(), format='%d.%m.%Y', errors='coerce')

table2_time['date_confirmation'] = pd.to_datetime(table2_time['date_confirmation'].str.strip(), format='%d.%m.%Y', errors='coerce')



# print(table2.columns)

table2_time.head(2)
# Looking for which country has how much data (important to understand and interpret results)

print(table2['country'].value_counts()[:5])
# temp = table1.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max().reset_index()

# temp[temp['Confirmed'] == max(temp['Confirmed'])]



# OR

temp = table1_latest_gp[table1_latest_gp['Confirmed'] == max(table1_latest_gp['Confirmed'])]

print(temp['Country/Region'])

temp
temp = table1.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

(temp.style.background_gradient(cmap='Pastel1'))
temp_f = table1_latest_gp.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
import plotly as py

import plotly.graph_objects as go

import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



fig = go.Figure()

fig.add_trace(go.Scatter(

                x=table1.Date,

                y=table1['Confirmed'],

                name="Confirmed",

                line_color='blue'))



fig.add_trace(go.Scatter(

                x=table1.Date,

                y=table1['Recovered'],

                name="Recovered",

                line_color='red'))

fig.update_layout(title_text='Rate of cases over time (Time Series with Rangeslider)',

                  xaxis_rangeslider_visible=True)

py.offline.iplot(fig)
# Top 3 regions in china where confirmed cases are recorded

table1[table1['Country/Region'] == 'China'].groupby('Province/State').max()['Confirmed'].sort_values(ascending=False)[:3]
import plotly.express as px

# import plotly.graph_objects as go

temp = table1[table1['Country/Region'] == 'China'].groupby('Province/State').max().reset_index()

hubei = temp[temp['Province/State'] == 'Hubei']

hubei['color'] = 'red'

temp = temp[temp['Province/State'] != 'Hubei']

fig = px.scatter_geo(lat='Lat', lon='Long', data_frame=temp, size='Confirmed')

fig.add_trace(px.scatter_geo(lat='Lat', lon='Long', data_frame=hubei, size='Confirmed', color_discrete_sequence=['red']).data[0])

# fig = px.scatter_geo(lat='Lat', lon='Long', data_frame=hubei, size='Confirmed')

fig.update_layout(

        title = 'Spread in China - Hubei(red) ',

        geo = dict(

            lonaxis =dict(range=[72,135]),

            lataxis =dict(range=[17,50]),

            showcountries=True,

            countrycolor="black",

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            countrywidth = 0.8,

        ),

    )

fig.show()
plt.figure(figsize=(12, 7))

ax = plt.plot(table1.groupby('Date').sum().reset_index()[cases], linewidth=3)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('total Cases', size=20)

plt.legend(cases, prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
top_5 = list(temp_f['Country/Region'][:5])

plt.figure(figsize=(12,8))

plt.plot(table1[table1['Country/Region'] == top_5[0]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[1]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[2]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[3]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[4]].groupby('Date').sum().reset_index()['Confirmed'], linewidth=3)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('total Cases', size=20)

plt.legend(top_5, prop={'size': 15})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
table2_time_china = table2_time[table2_time['country'] == 'China']

time_to_hospital = pd.DataFrame((table2_time_china['date_admission_hospital'] - table2_time_china['date_onset_symptoms']).astype(int))

time_to_confirm = pd.DataFrame((table2_time_china['date_confirmation'] - table2_time_china['date_onset_symptoms']).astype('timedelta64[D]'))



print('Average number of days for china = '+ str(np.mean(np.array(time_to_confirm[time_to_confirm[0].between(0,500)]))))

plt.figure(figsize=(16,10))



sns.countplot(y=0, data = time_to_confirm[time_to_confirm[0].between(1,500) ], orient='v')

plt.xlabel('count', size=20)

plt.ylabel('# of days from first symptom to tested positive(CHINA)', size=20)

plt.show()
table2_time_korea = table2_time[table2_time['country'] == 'South Korea']

time_to_hospital = pd.DataFrame((table2_time_korea['date_admission_hospital'] - table2_time_korea['date_onset_symptoms']).astype(int))

time_to_confirm = pd.DataFrame((table2_time_korea['date_confirmation'] - table2_time_korea['date_onset_symptoms']).astype('timedelta64[D]'))



print('Average number of days for china = '+ str(np.mean(np.array(time_to_confirm[time_to_confirm[0].between(0,500)]))))

plt.figure(figsize=(16,10))



sns.countplot(y=0, data = time_to_confirm[time_to_confirm[0].between(1,500) ], orient='v')

plt.xlabel('count', size=20)

plt.ylabel('# of days from first symptom to tested positive(S. KOREA)', size=20)

plt.show()
time_to_hospital = pd.DataFrame((table2_time['date_admission_hospital'] - table2_time['date_onset_symptoms']).astype(int))

time_to_confirm = pd.DataFrame((table2_time['date_confirmation'] - table2_time['date_onset_symptoms']).astype('timedelta64[D]'))



print('Average number of days for china = '+ str(np.mean(np.array(time_to_confirm[time_to_confirm[0].between(0,500)]))))
temp = table1.groupby('Date').max().reset_index()

dates = temp.keys()

mortality_rate = []

recovery_rate = [] 



for i,row in temp.iterrows():

    confirmed_sum = temp.iloc[i]['Confirmed']

    death_sum = temp.iloc[i]['Deaths']

    recovered_sum = temp.iloc[i]['Recovered']



    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)





plt.figure(figsize=(12, 7))

plt.plot(mortality_rate, linewidth=3)

plt.plot(recovery_rate, linewidth=3)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('rate (0-1)', size=20)

plt.legend(['mortality rate', 'recovery rate'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
# TOP 10 COUNTRIES WITH MOST NUMBER OF DEATHS



temp_f = table1.groupby('Country/Region').max().reset_index()[['Country/Region','Deaths','Recovered']].sort_values('Deaths', ascending=False)[:10].reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
top_5 = temp_f['Country/Region'][:6]

temp = table1[table1['Country/Region'] == 'US'].groupby('Date').max().reset_index()



plt.figure(figsize=(12,8))

plt.plot(table1[table1['Country/Region'] == top_5[0]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[1]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[2]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[3]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[4]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)

plt.plot(table1[table1['Country/Region'] == top_5[5]].groupby('Date').max().reset_index()['Deaths'], linewidth=3)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('total deaths', size=20)

plt.legend(top_5, prop={'size': 15})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
# import re

# tempr = table2.drop(['latitude', 'longitude', 'wuhan(0)_not_wuhan(1)','date_onset_symptoms','date_admission_hospital','date_confirmation', 'lives_in_Wuhan',

#                      'sequence_available','notes_for_discussion', 'chronic_disease', 'outcome','date_death_or_discharge', 'chronic_disease_binary', 

#                      'reported_market_exposure'], axis=1)



# italy = tempr[tempr['country'] == 'Italy'][['country','travel_history_location']]

# for index,row in tempr.iterrows():

#     if(row['travel_history_location']) != (row['travel_history_location']):

#         row['travel_history_location'] = np.nan

#     else:

#         if re.findall(r"Wuhan", row['travel_history_location']) or re.findall(r"wuhan", row['travel_history_location']):

#             print(row['travel_history_location'])

#             tempr.loc[index, 'travel_history_location'] = 'Wuhan'

#             print(row['travel_history_location'])

#             print('\n')



# D = tempr['travel_history_location'].value_counts().to_dict()

# plt.bar(range(len(D)), list(D.values()))

# plt.xticks(range(len(D)), list(D.keys()), rotation=90)

# plt.show()