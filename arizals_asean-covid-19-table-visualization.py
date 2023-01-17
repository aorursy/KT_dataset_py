# import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")
day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')
last_update = day_wise['Date'].max()
print('Last Update: {}'.format(last_update))
country_latest = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')
country_latest = country_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
country_latest.sort_values(by='Confirmed', ascending=False)\
                        .style.background_gradient(cmap='YlOrBr',subset=["Confirmed"])\
                        .background_gradient(cmap='Reds',subset=["Deaths"])\
                        .background_gradient(cmap='Greens',subset=["Recovered"])\
                        .background_gradient(cmap='Purples',subset=["Active"])\
                        .format("{:.0f}",subset=['Confirmed', 'Deaths', 'Recovered', 'Active'])
country_latest = pd.read_csv('/kaggle/input/corona-virus-report/country_wise_latest.csv')
asean_latest = country_latest[country_latest['Country/Region'].isin(['Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Burma', 'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam'])]
asean_latest['Country/Region'] = asean_latest['Country/Region'].replace('Burma','Myanmar')
asean_latest
asean_cases = asean_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
asean_cases.style.background_gradient(cmap='YlOrBr',subset=["Confirmed"])\
                        .background_gradient(cmap='Reds',subset=["Deaths"])\
                        .background_gradient(cmap='Greens',subset=["Recovered"])\
                        .background_gradient(cmap='Purples',subset=["Active"])\
                        .format("{:.0f}",subset=['Confirmed', 'Deaths', 'Recovered', 'Active'])
asean_confirmed = asean_cases.sort_values(by='Confirmed', ascending=False)
asean_confirmed.style.background_gradient(cmap='YlOrBr',subset=['Confirmed'])\
                .format("{:.0f}",subset=['Confirmed', 'Deaths', 'Recovered', 'Active'])
asean_deaths = asean_cases.sort_values(by='Deaths', ascending=False)
asean_deaths.style.background_gradient(cmap='Reds',subset=['Deaths'])\
            .format("{:.0f}",subset=['Confirmed', 'Deaths', 'Recovered', 'Active'])
asean_recovered = asean_cases.sort_values(by='Recovered', ascending=False)
asean_recovered.style.background_gradient(cmap='Greens',subset=['Recovered'])\
                .format("{:.0f}",subset=['Confirmed', 'Deaths', 'Recovered', 'Active'])
asean_active = asean_cases.sort_values(by='Active', ascending=False)
asean_active.style.background_gradient(cmap='Purples',subset=['Active'])\
            .format("{:.0f}",subset=['Confirmed', 'Deaths', 'Recovered', 'Active'])
asean_new_cases = asean_latest.groupby('Country/Region')['New cases', 'New deaths', 'New recovered'].sum()
asean_new_cases.sort_values(by='New cases', ascending=False)\
                        .style.background_gradient(cmap='YlOrBr',subset=['New cases'])\
                        .background_gradient(cmap='Reds',subset=['New deaths'])\
                        .background_gradient(cmap='Greens',subset=['New recovered'])