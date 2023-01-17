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
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

import itertools

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import folium



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/US_daily_cases.csv')
##Drop the unimportant columns

df = df.drop(['Country_Region','Last_Update','ISO3','UID'], axis=1)
us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS',

    'Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE',

    'District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID',

    'Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA',

    'Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS',

    'Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ',

    'New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Northern Mariana Islands':'MP','Ohio': 'OH',

    'Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC',

    'South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI',

    'Virginia': 'VA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'

}
df['state_code'] = df["Province_State"].map(us_state_abbrev)
usa_pop = pd.read_csv('/kaggle/input/us_statewise_population.csv')

state_pop = usa_pop.set_index('State').to_dict()['Pop']

df['state_pop'] = df["Province_State"].map(state_pop)


columns_with_na = list(df.columns[df.isna().any()])  #For geting list of column with NAN

print(columns_with_na)
for col in columns_with_na:

    if df[col].dtype != 'object':

        df.update(df[col].fillna(0))
df_top15 = df[['Province_State','state_code','Confirmed','Deaths','Active','Recovered','state_pop']].sort_values(['Confirmed'], ascending = False).head(15)
f, ax = plt.subplots(figsize = (20,10), ncols=3)

sns.color_palette("pastel")

f = sns.barplot(x = 'Confirmed', y = 'state_code', 

            data = df_top15[['state_code','Confirmed']],

        label = 'Confirmed', color = 'r', edgecolor = 'w', ax=ax[0])

sns.barplot(x = 'Deaths', y = 'state_code', 

            data = df_top15[['state_code','Deaths']],

        label = 'Deaths', color = 'gray', edgecolor = 'w', ax=ax[0])

ax[0].set_title('Confirmed Vs Deaths')

ax[0].legend(ncol = 1, loc = 'bottom right')

for item in f.get_xticklabels(): item.set_rotation(45)

    

f = sns.barplot(x = 'Confirmed', y = 'state_code', 

            data = df_top15[['state_code','Confirmed']],

        label = 'Confirmed', color = 'r', edgecolor = 'w', ax=ax[1])

sns.barplot(x = 'Active', y = 'state_code', 

            data = df_top15[['state_code','Active']],

        label = 'Active', color = 'b', edgecolor = 'w', ax=ax[1])

ax[1].set_title('Confirmed vs Active')

ax[1].legend(ncol = 1, loc = 'bottom right')

for item in f.get_xticklabels(): item.set_rotation(45)

    

f = sns.barplot(x = 'Confirmed', y = 'state_code', 

            data = df_top15[['state_code','Confirmed']],

        label = 'Confirmed', color = 'r', edgecolor = 'w', ax=ax[2])

sns.barplot(x = 'Recovered', y = 'state_code', 

            data = df_top15[['state_code','Recovered']],

        label = 'Recovered', color = 'g', edgecolor = 'w', ax=ax[2])

ax[2].set_title('Confirmed Vs Recovered')

ax[2].legend(ncol = 1, loc = 'bottom right')

for item in f.get_xticklabels(): item.set_rotation(45)



sns.despine(left = True, bottom = True)

plt.show()
df_million = df[['state_code','Confirmed','Deaths','Recovered','state_pop']].where(df['state_pop'] > 0).dropna()

df_million['confirmed_million'] = (df_million.Confirmed/(df_million.state_pop/1000000)).astype(int)

df_million['deaths_million'] = (df_million.Deaths/(df_million.state_pop/1000000)).astype(int)

df_million['recovered_million'] = (df_million.Recovered/(df_million.state_pop/1000000)).astype(int)



f, ax = plt.subplots(figsize = (20,10), ncols=3)

sns.color_palette("pastel")

sns.barplot(x = 'confirmed_million', y = 'state_code', 

            data = df_million[['state_code','confirmed_million']].sort_values(['confirmed_million'], ascending = False).head(15),

        label = 'Confirmed', color = 'r', edgecolor = 'w', ax=ax[0])

ax[0].set_title('Confirmed Cases/Million')

ax[0].legend(ncol = 1, loc = 'bottom right')

sns.barplot(x = 'deaths_million', y = 'state_code', 

            data = df_million[['state_code','deaths_million']].sort_values(['deaths_million'], ascending = False).head(15),

        label = 'Deaths', color = 'gray', edgecolor = 'w', ax=ax[1])

ax[1].set_title('Deaths/Million')

ax[1].legend(ncol = 1, loc = 'bottom right')

sns.barplot(x = 'recovered_million', y = 'state_code', 

            data = df_million[['state_code','recovered_million']].sort_values(['recovered_million'], ascending = False).head(15),

        label = 'recovered_million', color = 'g', edgecolor = 'w', ax=ax[2])

ax[2].set_title('Recovered/Million')

ax[2].legend(ncol = 1, loc = 'bottom right')

sns.despine(left = True, bottom = True)

plt.show()
f, ax = plt.subplots(figsize = (9,9))

cmap = sns.cubehelix_palette(as_cmap = True, dark = 0, light = 1, reverse = True)

sns.kdeplot(df_top15['Confirmed'][2:], df_top15['Deaths'][2:], cmap = cmap, shade=True)
global_daily = pd.read_csv('/kaggle/input/daily_cases.csv')

usa_daily = global_daily[global_daily['Country_Region'] == 'US']
fig = px.scatter_mapbox(usa_daily[usa_daily['Confirmed'] > 0], lat="Lat", lon="Long_", hover_name="Confirmed", hover_data=["Combined_Key", "Confirmed"],

                        color_discrete_sequence=["red"],zoom = 3, height = 350)



fig.update_layout(mapbox_style="open-street-map",

                  title_text = 'Confirmed Cases',

                 margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.scatter_mapbox(usa_daily[usa_daily['Deaths'] > 0], lat="Lat", lon="Long_", hover_name="Confirmed", hover_data=["Combined_Key", "Deaths"],

                        color_discrete_sequence=["gray"],zoom = 2.9, height = 350)



fig.update_layout(mapbox_style="open-street-map",

                  title_text = 'Deaths',

                 margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.scatter_mapbox(usa_daily[usa_daily['Active'] > 0], lat="Lat", lon="Long_", hover_name="Active", hover_data=["Combined_Key", "Active"],

                        color_discrete_sequence=["orange"],zoom = 2.9, height = 350)



fig.update_layout(mapbox_style="open-street-map",

                  title_text = 'Active Cases',

                 margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.scatter_mapbox(df[df['Recovered'] > 0], lat="Lat", lon="Long_", hover_name="Recovered", hover_data=["state_code", "Recovered"],

                        color_discrete_sequence=["green"],zoom = 2.9, height = 350,size = 'Recovered')





fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()