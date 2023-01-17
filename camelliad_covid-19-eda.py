# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_recovered= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

time_series_covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

covid_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

COVID19_open_line = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

COVID19_line_list_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
covid_data.head()
top_countries= covid_data.groupby('Country/Region').max().sort_values(by='Deaths', ascending=False)[:10]

top_countries.head()

top_countries['Country/Region'] = top_countries.index
print('Top 10 worst affected countries in terms of death toll')

top_countries
plt.figure(figsize=[10,8])

plt.xlabel('#cases', color="white")

plt.ylabel('Countries', color="white")

plt.title('Number of Covid-19 cases in the 10 most affected countries')

plt.xticks(rotation=45)

plt.bar(top_countries['Country/Region'],top_countries['Confirmed'], label='Confirmed')

plt.bar(top_countries['Country/Region'],top_countries['Recovered'], color="orange", label='Recovered')

plt.bar(top_countries['Country/Region'],top_countries['Deaths'], color="red", label='Deaths')

plt.legend()

plt.grid()
covid_china = covid_data[covid_data['Country/Region']=='Mainland China']

covid_china = covid_china.groupby('ObservationDate').max()

covid_china['ObservationDate'] = covid_china.index



covid_italy = covid_data[covid_data['Country/Region']=='Italy']

covid_italy = covid_italy.groupby('ObservationDate').max()

covid_italy['ObservationDate'] = covid_italy.index



covid_iran = covid_data[covid_data['Country/Region']=='Iran']

covid_iran = covid_iran.groupby('ObservationDate').max()

covid_iran['ObservationDate'] = covid_iran.index



covid_spain = covid_data[covid_data['Country/Region']=='Spain']

covid_spain = covid_spain.groupby('ObservationDate').max()

covid_spain['ObservationDate'] = covid_spain.index



covid_france = covid_data[covid_data['Country/Region']=='France']

covid_france = covid_france.groupby('ObservationDate').max()

covid_france['ObservationDate'] = covid_france.index



covid_us = covid_data[covid_data['Country/Region']=='US']

covid_us = covid_us.groupby('ObservationDate').max()

covid_us['ObservationDate'] = covid_us.index

x = covid_china['ObservationDate'].tolist()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,15))

fig.tight_layout()





axes[0].plot(covid_china['ObservationDate'],covid_china['Confirmed'], color = 'green', label='China', marker='o')

axes[0].plot(covid_italy['ObservationDate'],covid_italy['Confirmed'], color="red", label='Italy',marker='o')

axes[0].plot(covid_iran['ObservationDate'],covid_iran['Confirmed'], color="blue", label='Iran',marker='o')

axes[0].plot(covid_spain['ObservationDate'],covid_spain['Confirmed'], color="cyan", label='Spain',marker='o')

axes[0].plot(covid_france['ObservationDate'],covid_france['Confirmed'], color="purple", label='France',marker='o')

axes[0].plot(covid_us['ObservationDate'],covid_us['Confirmed'], color="salmon", label='US',marker='o')

axes[0].set_xlabel('Dates'+"\n\n")

axes[0].set_ylabel('Cases')

axes[0].set_title('Number of Covid-19 confirmed cases, 1/22 - 3/19')

axes[0].set_xticks(axes[0].get_xticks()[::5])

axes[0].set_xticklabels(x[::5], rotation=45)

axes[0].grid()

axes[0].legend(loc="upper left")

plt.subplots_adjust(hspace = 0.4)





axes[1].plot(covid_china['ObservationDate'],covid_china['Deaths'], color = 'green', label='China',marker='o')

axes[1].plot(covid_italy['ObservationDate'],covid_italy['Deaths'], color="red", label='Italy',marker='o')

axes[1].plot(covid_iran['ObservationDate'],covid_iran['Deaths'], color="blue", label='Iran',marker='o')

axes[1].plot(covid_spain['ObservationDate'],covid_spain['Deaths'], color="cyan", label='Spain',marker='o')

axes[1].plot(covid_france['ObservationDate'],covid_france['Deaths'], color="purple", label='France',marker='o')

axes[1].plot(covid_us['ObservationDate'],covid_us['Deaths'], color="salmon", label='US',marker='o')

axes[1].set_xlabel('Dates\n\n')

axes[1].set_ylabel('Cases')

axes[1].set_title('Number of Covid-19 Deaths, 1/22 - 3/19')

axes[1].set_xticks(axes[1].get_xticks()[::5])

axes[1].set_xticklabels(x[::5], rotation=90)

axes[1].grid()

axes[1].legend(loc="upper left")

plt.subplots_adjust(hspace = 0.4)



axes[2].plot(covid_china['ObservationDate'],covid_china['Recovered'], color = 'green', label='China',marker='o')

axes[2].plot(covid_italy['ObservationDate'],covid_italy['Recovered'], color="red", label='Italy',marker='o')

axes[2].plot(covid_iran['ObservationDate'],covid_iran['Recovered'], color="blue", label='Iran',marker='o')

axes[2].plot(covid_spain['ObservationDate'],covid_spain['Recovered'], color="cyan", label='Spain',marker='o')

axes[2].plot(covid_france['ObservationDate'],covid_france['Recovered'], color="purple", label='France',marker='o')

axes[2].plot(covid_us['ObservationDate'],covid_us['Recovered'], color="salmon", label='US',marker='o')

axes[2].set_xlabel('Dates'+"\n\n")

axes[2].set_ylabel('Cases')

axes[2].set_title('Number of Covid-19 Recoveries, 1/22 - 3/19')

axes[2].set_xticks(axes[2].get_xticks()[::5])

axes[2].set_xticklabels(x[::5], rotation=90)

axes[2].grid()

axes[2].legend(loc="upper left")
import sys

from sys import argv

import requests

import datetime

import pandas as pd

from bs4 import BeautifulSoup

from tabulate import tabulate



url = 'https://www.worldometers.info/coronavirus/'



class HTMLTableParser:



    def parse_url(self, url):

        response = requests.get(url)

        soup = BeautifulSoup(response.text, 'lxml')

        return [(table['id'],self.parse_html_table(table))\

            for table in soup.find_all('table')]



    def parse_html_table(self, table):

        n_columns = 0

        n_rows = 0

        column_names = []



        for row in table.find_all('tr'):

            td_tags = row.find_all('td')

            if len(td_tags) > 0:

                n_rows += 1

                if n_columns == 0:

                    n_columns = len(td_tags)

        

        th_tags = row.find_all('th')

        if len(th_tags) > 0 and len(column_names) == 0:

            for th in th_tags:

                column_names.append(th.get_text())

        

        if len(column_names) > 0 and len(column_names) != n_columns:

            raise Exception("Column titles do not match the number of columns")



        columns = column_names if len(column_names) > 0 else range(0,n_columns)

        df = pd.DataFrame(columns = columns,

                    index = range(0,n_rows))

        row_marker = 0

        for row in table.find_all('tr'):

            column_marker = 0

            columns = row.find_all('td')

            for column in columns:

                df.iat[row_marker,column_marker] = column.get_text()

                column_marker += 1

            if len(columns) > 0:

                row_marker += 1



        for col in df:

            try:

                df[col] = df[col].astype(float)

            except ValueError:

                pass

        

        return df

    

time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M%Z")

print("\n" + "Date/Time >: " + time)

print("Counters are reset at 23:59UTC" + "\n")



hp = HTMLTableParser()

table = hp.parse_url(url)[0][1]

table.columns = ["Country", "Cases","New Cases","Deaths","New Deaths","Recovered","Active","Critical","cases/million"]

table.head()