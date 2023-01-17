# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import plotly.graph_objects as go
covid_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

covid_data.head()
covid_data.info()
print("Starting date : ", min(covid_data.ObservationDate.values))

print("Ending date : ", max(covid_data.ObservationDate.values))
covid_data.ObservationDate = pd.to_datetime(covid_data.ObservationDate)

tot_rates = covid_data.groupby('ObservationDate').sum()

tot_rates.head()
fig = go.Figure()



fig.add_trace(go.Scatter(x = tot_rates.index, y = tot_rates.Confirmed, name = 'Confirmed'))

fig.add_trace(go.Scatter(x = tot_rates.index, y = tot_rates.Recovered, name = 'Recovered'))

fig.add_trace(go.Scatter(x = tot_rates.index, y = tot_rates.Deaths, name = 'Deaths'))

fig.update_layout(title = 'COVID-19 CASES ALL OVER THE WORLD', xaxis_title='Time (Jan 2020 - Aug 2020)',

                   yaxis_title='Count of Cases')

fig.show()
# Since the data is cumulative so, we're taking only the last date data.



country = covid_data[covid_data.ObservationDate == max(covid_data.ObservationDate)]

cntry_case = country.groupby('Country/Region').sum()

cntry_case
cntry_case['recover_rate'] = (cntry_case.Recovered/cntry_case.Confirmed)*100

cntry_case['death_rate'] = (cntry_case.Deaths/cntry_case.Confirmed)*100



cntry_case.head()
Confirmed_df = cntry_case.sort_values(by=['Confirmed'], ascending=False)

Recovered_df = cntry_case.sort_values(by=['Recovered'], ascending=False)

Death_df = cntry_case.sort_values(by=['Deaths'], ascending=False)

Recover_rate_df = cntry_case.sort_values(by=['recover_rate'], ascending=False)

Death_rate_df = cntry_case.sort_values(by=['death_rate'], ascending=False)
fig = go.Figure()

fig.add_trace(go.Bar(x = Confirmed_df.index[:15], y = Confirmed_df['Confirmed'][:15], name='Confirmed'))

fig.add_trace(go.Bar(x = Confirmed_df.index[:15], y = Confirmed_df['Recovered'][:15], name='Recovered'))

fig.add_trace(go.Bar(x = Confirmed_df.index[:15], y = Confirmed_df['Deaths'][:15], name='Deaths'))

fig.update_layout(title='15 Worst Corona-Virus hit countries uptill now', xaxis_title='Countries',

                 yaxis_title='Counts of Cases (in millions)')

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x = Confirmed_df.index[-15:], y = Confirmed_df['Confirmed'][-15:], name='Confirmed'))

fig.add_trace(go.Bar(x = Confirmed_df.index[-15:], y = Confirmed_df['Recovered'][-15:], name='Recovered'))

fig.add_trace(go.Bar(x = Confirmed_df.index[-15:], y = Confirmed_df['Deaths'][-15:], name='Deaths'))

fig.update_layout(title='15 Less Corona-Virus hit countries uptill now', xaxis_title='Countries',

                 yaxis_title='Counts of Cases (in millions)')

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x = Recovered_df.index[:15], y = Recovered_df['Confirmed'][:15], name='Confirmed'))

fig.add_trace(go.Bar(x = Recovered_df.index[:15], y = Recovered_df['Recovered'][:15], name='Recovered'))

fig.add_trace(go.Bar(x = Recovered_df.index[:15], y = Recovered_df['Deaths'][:15], name='Deaths'))

fig.update_layout(title='Top 15 countries in recovering rate', xaxis_title='Countries',

                 yaxis_title='Cases (in millions)')

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x = Recovered_df.index[-15:], y = Recovered_df['Confirmed'][-15:], name='Confirmed'))

fig.add_trace(go.Bar(x = Recovered_df.index[-15:], y = Recovered_df['Recovered'][-15:], name='Recovered'))

fig.add_trace(go.Bar(x = Recovered_df.index[-15:], y = Recovered_df['Deaths'][-15:], name='Deaths'))

fig.update_layout(title="Last 15 countries in recovered cases", xaxis_title='Countries',

                 yaxis_title='Cases (in millions)')

fig.show()
last15 = Recovered_df[(Recovered_df.index != 'Sweden') & (Recovered_df.index != 'Serbia')][-15:]



fig = go.Figure()

fig.add_trace(go.Bar(x = last15.index, y = last15['Confirmed'], name='Confirmed'))

fig.add_trace(go.Bar(x = last15.index, y = last15['Recovered'], name='Recovered'))

fig.add_trace(go.Bar(x = last15.index, y = last15['Deaths'], name='Deaths'))

fig.update_layout(title="Last 15 countries in recovered cases", xaxis_title='Countries',

                 yaxis_title='Cases (in millions)')

fig.show()
fig = go.Figure(data=go.Bar(x = Death_df.index[:15], y = Death_df['Deaths'][:15]))

fig.update_layout(title="Top 15 countries in death cases", xaxis_title='Countries',

                 yaxis_title='Cases (in millions)')

fig.show()
fig = go.Figure(data=go.Bar(x = Death_df.index[-15:], y = Death_df['Deaths'][-15:]))

fig.update_layout(title="Last 15 countries in death cases", xaxis_title='Countries',

                 yaxis_title='Cases (in millions)')

fig.show()
India_df = covid_data[covid_data['Country/Region'] == 'India']

India_df = India_df.groupby('ObservationDate').sum()

    

fig = go.Figure()

fig.add_trace(go.Scatter(x = India_df.index, y = India_df.Confirmed, name = 'Confirmed'))

fig.add_trace(go.Scatter(x = India_df.index, y = India_df.Recovered, name = 'Recovered'))

fig.add_trace(go.Scatter(x = India_df.index, y = India_df.Deaths, name = 'Deaths'))

fig.update_layout(title='Increase of COVID-19 Cases of India', xaxis_title='Time',

                  yaxis_title='Count in millions')



fig.show()
fig = go.Figure(data = go.Pie(labels = cntry_case.index, values = cntry_case.Confirmed,

                name = 'Pie chart of countries'))

fig.update_layout(title = 'Pie chart showing distribution of corona virus cases of different countries')

fig.show()
India_df = covid_data[covid_data['Country/Region'] == 'India']

India_df = India_df.groupby('ObservationDate').sum()

India_df['Recover_rate'] = (India_df.Recovered/India_df.Confirmed)*100

India_df['Death_rate'] = (India_df.Deaths/India_df.Confirmed)*100



fig = go.Figure(data = go.Scatter(x = India_df.index, y = India_df.Recover_rate, name = 'Recovery Rate'))

fig.update_layout(title = 'Graph of Recovery rate', xaxis_title = 'Time', yaxis_title = "Recovery rate in percentage(%)")

fig.show()



fig = go.Figure(data = go.Scatter(x = India_df.index, y = India_df.Death_rate, name = 'Death Rate'))

fig.update_layout(title = 'Graph of Death Rate', xaxis_title = 'Time', yaxis_title = "Death rate in percentage(%)")

fig.show()