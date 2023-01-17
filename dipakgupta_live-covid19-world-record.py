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
from bs4 import BeautifulSoup

import requests

import pandas as pd

import numpy as np
url="https://www.worldometers.info/coronavirus/"

html_content = requests.get(url).text

soup = BeautifulSoup(html_content, "lxml")

gdp_table = soup.find("table", id = "main_table_countries_today")

gdp_table_data = gdp_table.tbody.find_all("tr")



samp = {}

for i in range(len(gdp_table_data)):

    try:

        key = (gdp_table_data[i].find_all('a', href=True)[0].string)

    except:

        key = (gdp_table_data[i].find_all('td')[0].string)

    value = [j.string for j in gdp_table_data[i].find_all('td')]

    samp[key] = value

live_data= pd.DataFrame(samp).drop(0).T.iloc[:,:12]

live_data.columns = ["Country","Total Cases","New Cases", "Total Deaths", "New Deaths", "Total Recovered","dff","Active","Serious Critical",

"Tot Cases/1M pop","Deaths/1M pop","Total Test"]

live_data.index.name = 'ctr'

live_data.to_csv("./base_data.csv")
data = pd.read_csv("./base_data.csv")
data.drop(['dff','ctr','Tot Cases/1M pop','Deaths/1M pop'],inplace=True,axis=1)
import json

countryy = json.load(open("../input/live-covid19-world-records/countries.geojson",'r'))
data2 = data.copy()
lst_name = ['World','DRC','Eswatini','CAR','Guadeloupe','Mayotte','Réunion','Martinique','Diamond Princess','Channel Islands',

            'Faeroe Islands','Curaçao','Caribbean Netherlands','Timor-Leste','MS Zaandam']

for i in lst_name:

    data2.drop(data2[data2['Country'] == i].index,inplace=True)
map_iso_alpha = {}

for states_prop in countryy['features']:

    states_prop['iso_alpha'] = states_prop['properties']['ISO_A3']

    map_iso_alpha[states_prop['properties']['ADMIN']] = states_prop['iso_alpha']
data['iso_alpha'] = data2['Country'].apply(lambda x: map_iso_alpha[x])
import plotly.express as px
fig = px.choropleth(data, locations="iso_alpha",

                    color="Total Cases", # lifeExp is a column of gapminder

                    hover_name="Country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
data['total_case'] = data['Total Cases'].apply(lambda x: int(x.replace(",","")))
data['Total_Cases_scale'] = np.log10(data['total_case'])
fig = px.choropleth(data, locations = "iso_alpha",

                    color = "Total_Cases_scale", # lifeExp is a column of gapminder

                    hover_name = "Country", # column to add to hover information

                    hover_data = ["Total Cases"],

                    color_continuous_scale = px.colors.sequential.Plasma)

fig.show()
#data['Total_Deaths'] = data['Total Deaths'].apply(lambda x: x.replace(",",""))

#data['Total_Deaths'] = data['Total_Deaths'].apply(lambda x: int(x.replace(" ",str('0')))/10)
data['Total_Deaths'] = data['Total Deaths'].apply(lambda x: x.split(" ")[0].replace(",",""))
data['Total_Deaths'] = data['Total_Deaths'].apply(lambda x: int(x) if len(x)>0 else 0)
data['Total_Deaths_scale'] = data['Total_Deaths'].apply(lambda x: np.log10(x) if x >= 1 else 0)
#data['Total_Deaths_scale'] = np.log10(data['Total_Deaths'])
import plotly.express as px

fig = px.choropleth(data, locations = "iso_alpha",

                    color = "Total_Deaths_scale", # lifeExp is a column of gapminder

                    hover_name = "Country", # column to add to hover information

                    hover_data = ["Total Deaths"],

                    color_continuous_scale = px.colors.sequential.Plasma)

fig.show()
import matplotlib.pyplot as plt

import seaborn as sns
data_case_sort = data.sort_values(["total_case"], axis=0, 

                 ascending=False)
fig = plt.figure()

plt.rcParams['figure.figsize'] = 25,10

sns.set(style="whitegrid")

ss = sns.barplot(y=data_case_sort['total_case'].head(20),x=data_case_sort['Country'].head(20))

for p in ss.patches:

    ss.annotate(format(p.get_height(), '.0f'), 

               (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', 

               xytext = (0, 10), 

               textcoords = 'offset points')

fig.suptitle('BARPLOT', fontsize=15)

plt.xticks(rotation=45,fontsize=15)

plt.yticks(fontsize=15)

plt.ylabel('Total cases', fontsize=18)

plt.xlabel("Country", fontsize=18)
data_death_sort = data.sort_values(["Total_Deaths"], axis=0, 

                 ascending=False)
plt.rcParams['figure.figsize'] = 25,10

fig = plt.figure()

sns.set(style="whitegrid")

ss = sns.barplot(y=data_death_sort['Total_Deaths'].head(20),x=data_death_sort['Country'].head(20))

for p in ss.patches:

    ss.annotate(format(p.get_height(), '.0f'), 

               (p.get_x() + p.get_width() / 2., p.get_height()), 

               ha = 'center', va = 'center', 

               xytext = (0, 10), 

               textcoords = 'offset points')

fig.suptitle('BARPLOT', fontsize=20)

plt.xticks(rotation=45,fontsize=15)

plt.yticks(fontsize=15)

plt.ylabel('Total Deaths', fontsize=18)

plt.xlabel("Country", fontsize=18)