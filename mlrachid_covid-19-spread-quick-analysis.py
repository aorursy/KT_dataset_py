# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')

data.head()
data.tail()
data.info()
#data.drop(['Longitude', 'Latitude'], axis=1, inplace=True)

data.dropna(subset=['Confirmed', 'Recovered', 'Deaths', 'Longitude', 'Latitude'], inplace=True)

data.info()
countries = data['Country/Region'].unique().tolist()

def mercator(data, lon="Longitude", lat="Latitude"):

    """Converts decimal longitude/latitude to Web Mercator format"""

    k = 6378137

    data["x"] = data[lon] * (k * np.pi/180.0)

    data["y"] = np.log(np.tan((90 + data[lat]) * np.pi/360.0)) * k

    return data



data = mercator(data)

data.head()
from bokeh.plotting import figure

from bokeh.io import output_notebook, show

from bokeh.models import WMTSTileSource

output_notebook()








url = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'

p = figure(tools='pan, wheel_zoom', x_axis_type="mercator", y_axis_type="mercator")



p.add_tile(WMTSTileSource(url=url))

p.circle(x=data['x'], y=data['y'], fill_color='orange', size=5)

show(p)
len(countries)
union_europe = ['Austria', 'Italy', 'Belgium', 'Latvia', 'Bulgaria', 'Lithuania', 'Croatia', 'Luxembourg',

                'Cyprus', 'Malta', 'Czechia', 'Netherlands', 'Denmark', 'Poland', 'Estonia', 'Portugal',

                'Finland', 'Romania', 'France', 'Slovakia', 'Germany', 'Slovenia', 'Greece', 'Spain',

                'Hungary', 'Sweden', 'Ireland', 'UK']
non_EU = ['Albania', 'Belarus', 'Bosnia', 'Herzegovina', 'Kosovo', 'Macedonia', 'Moldova',

          'Norway', 'Russia', 'Serbia', 'Switzerland', 'Ukraine', 'Turkey']
data_EU = data[data['Country/Region'].isin(union_europe)]

data_non_EU = data[data['Country/Region'].isin(non_EU)]
aisa_countries = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh' 'Bhutan',

                  'Brunei', 'Cambodia', 'Mainland China', 'Cyprus', 'Georgia','Hong Kong',

                  'India' 'Indonesia',

                  'Iran', 'Iraq', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan'

                  , 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar',

                  'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines',

                  'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka','Syria',

                  'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkmenistan',

                  'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']
data_aisa = data[data['Country/Region'].isin(aisa_countries)]
africa_countries = ['Liberia', 'Tanzania', 'Eritrea','Ethiopia', 'Cameroon', 'Ghana','South Africa', 'Kenya', 'Rwanda','Nigeria', 'Gabon', 'Tunisia','Senegal', 'Algeria', 'Ivory Coast','Uganda', 'Morocco', 'Zimbabwe','Egypt']
data_africa = data[data['Country/Region'].isin(africa_countries)]
america_countries = ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic',

                   'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico',

                   'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'US']
data_america = data[data['Country/Region'].isin(america_countries)]
data_australia = data[(data['Country/Region']=='New Zealand')|(data['Country/Region']=='Australia')]
data[data['Country/Region']=='Others'].shape
total_confirmed = [data_africa['Confirmed'].max(), data_aisa['Confirmed'].max(), data_EU['Confirmed'].max(), data_non_EU['Confirmed'].max(), data_america['Confirmed'].max(), data_australia['Confirmed'].max()]

total_deaths = [data_africa['Deaths'].max(), data_aisa['Deaths'].max(), data_EU['Deaths'].max(), data_non_EU['Deaths'].max(), data_america['Deaths'].max(), data_australia['Deaths'].max()]
areas = ['Africa', 'Aisa', 'EU', 'NON-EU', 'America', 'Australia']

df_continents = pd.DataFrame({'Confirmed':total_confirmed, 'Deaths':total_deaths}, index=areas)
df_continents
sns.set()

plt.figure(figsize=(12, 6), dpi=300)

position = np.arange(len(areas))

width = 0.4

plt.bar(position - (width/2), (df_continents['Confirmed']/df_continents['Confirmed'].sum())*100, width=width, label='Confirmed')

plt.bar(position + (width/2), (df_continents['Deaths']/df_continents['Deaths'].sum())*100, width=width, label='Deaths')

plt.xticks(position, rotation=10)

plt.yticks(np.arange(0, 101, 10))

ax = plt.gca()

ax.set_xticklabels(areas)

ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']);

ax.set_yticks(np.arange(0, 100, 5), minor=True)

ax.yaxis.grid(which='major')

ax.yaxis.grid(which='minor', linestyle='--')

plt.title('Confirmed vs Deaths in different continents')

plt.legend();

plt.figure(figsize=(10, 6))



plt.plot('Date', 'Confirmed', data=data_aisa, label='Aisa')

plt.plot('Date', 'Confirmed', data=data_EU, label='EU')

plt.plot('Date', 'Confirmed', data=data_non_EU, label='Non-EU')

plt.plot('Date', 'Confirmed', data=data_america, label='America')

plt.plot('Date', 'Confirmed', data=data_australia, label='Australia')

plt.xticks(np.arange(0, 60, 2), rotation=70)

plt.legend()

plt.title('Confirmed cases in different continents')
plt.figure(figsize=(10, 6))

plt.plot('Date', 'Deaths', data=data_aisa, label='Aisa')

plt.plot('Date', 'Deaths', data=data_EU, label='EU')

plt.plot('Date', 'Deaths', data=data_non_EU, label='Non-EU')

plt.plot('Date', 'Deaths', data=data_america, label='America')

plt.plot('Date', 'Deaths', data=data_australia, label='Australia')

plt.xticks(np.arange(0, 60, 2), rotation=70)

plt.yticks(np.arange(0, 3001, 500))

plt.legend()

plt.title('Deaths in different continents')