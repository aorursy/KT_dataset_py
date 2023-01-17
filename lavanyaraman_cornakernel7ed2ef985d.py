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
# Import Python Packages

import pandas as pd

import numpy as np

import datetime as dt

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
COVID19_line_list_data.head
time_series_covid_19_confirmed.head
covid_19_data.head
covid_19_data.shape
covid_19_data.head(50)
covid_19_data = covid_19_data.dropna(axis=0)
x=[]

y=[]

labels = covid_19_data['Country/Region']

count = covid_19_data['Confirmed']

plt.xticks(np.arange(0,35), covid_19_data['Province/State'], rotation=90)

x=list(labels)

y=list(count)

plt.title("covid-19 state wise data")

plt.xlabel("Province/State")

plt.ylabel("Confirmed")

plt.bar(x,y)

plt.show()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

covid_19_data['Active'] = covid_19_data['Confirmed'] - covid_19_data['Deaths'] - covid_19_data['Recovered']



# replacing Mainland china with just China

covid_19_data['Country/Region'] = covid_19_data['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

covid_19_data[['Province/State']] = covid_19_data[['Province/State']].fillna('')

covid_19_data[cases] = covid_19_data[cases].fillna(0)
covid_19_data.head(50)
# latest

covid_19_data_latest = covid_19_data[covid_19_data['ObservationDate'] == max(covid_19_data['ObservationDate'])].reset_index()

china_latest = covid_19_data_latest[covid_19_data_latest['Country/Region']=='China']

row_latest = covid_19_data_latest[covid_19_data_latest['Country/Region']!='China']

# latest condensed

covid_19_data_latest_grouped = covid_19_data_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
covid_19_data.head(25)
temp = covid_19_data.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()

temp.style.background_gradient(cmap='Reds')
temp = covid_19_data.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['ObservationDate']==max(temp['ObservationDate'])].reset_index(drop=True)

temp.style.background_gradient(cmap='autumn')
temp_f = covid_19_data_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='autumn')
temp_f = china_latest_grouped[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]

temp_f = temp_f.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')