!pip install plotly==3.10.0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization library

import matplotlib.pyplot as plt # visualization library

import plotly.plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object

from datetime import date, datetime, timedelta





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# import warnings library

import warnings        

# ignore filters

warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.

plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.

# Any results you write to the current directory are saved as output.
confirmed = pd.read_csv("../input/time_series_covid19_confirmed_global.csv")

deaths = pd.read_csv("../input/time_series_covid19_deaths_global.csv")

recovered = pd.read_csv("../input/time_series_covid19_recovered_global.csv")
confirmed.head()
confirmed.info()
print(confirmed.columns)

print(deaths.columns)

print(recovered.columns)
confirmed.columns = list(confirmed.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in confirmed.columns[4:]]

deaths.columns    = list(deaths.columns[:4])    + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in deaths.columns[4:]]

recovered.columns = list(recovered.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in recovered.columns[4:]]
confirmed.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

deaths.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)

recovered.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
confirmed.head()
dates = confirmed.columns[4:]



confirmed_long = confirmed.melt(

    id_vars=['Province_State', 'Country_Region', 'Lat', 'Long'], 

    value_vars=dates, 

    var_name='Date', 

    value_name='Confirmed'

)

deaths_long = deaths.melt(

    id_vars=['Province_State', 'Country_Region', 'Lat', 'Long'], 

    value_vars=dates, 

    var_name='Date', 

    value_name='Deaths'

)

recovered_long = recovered.melt(

    id_vars=['Province_State', 'Country_Region', 'Lat', 'Long'], 

    value_vars=dates, 

    var_name='Date', 

    value_name='Recovered'

)
confirmed_long
(confirmed_long['Country_Region'] == "Canada").value_counts()
recovered_long[recovered_long['Country_Region']=='Canada']
recovered_long = recovered_long[recovered_long['Country_Region']!='Canada']
# Check if there is Cana

(recovered_long['Province_State']=='Canada').value_counts()
# Merging 1: create full_table using confirmed_long and left joined by deaths_long

full_table = confirmed_long.merge(

    right = deaths_long,

    how = 'left',

    on = ['Province_State', 'Country_Region', 'Date', 'Lat', 'Long']

)



# Merging 2: merge full_table with recovered_long

full_table = full_table.merge(

    right = recovered_long,

    how = 'left',

    on = ['Province_State', 'Country_Region', 'Date', 'Lat', 'Long']

)
full_table
# Date is in datetime format

full_table.dtypes
# Convert Date from String to DateTime

#full_table['Date'] = pd.to_datetime(full_table['Date'])

#full_table.dtypes
full_table.isna().sum()
full_table['Recovered'] = full_table['Recovered'].fillna(0)

full_table[full_table['Recovered']==0]
full_table.info()
full_table['Recovered'] = full_table['Recovered'].astype('int')

full_table.info()
lst = ['Grand Princess', 'Diamond Princess', 'MS Zaandam']

lst = '|'.join(lst)

full_table.loc[full_table['Province_State'].str.contains(lst, na=False) | full_table['Country_Region'].str.contains(lst, na=False)]
ship_rows = full_table['Province_State'].str.contains('Grand Princess') | full_table['Province_State'].str.contains('Diamond Princess') | full_table['Country_Region'].str.contains('Diamond Princess') | full_table['Country_Region'].str.contains('MS Zaandam')

full_ship = full_table[ship_rows]

full_ship
full_table = full_table[~(ship_rows)]

full_table.loc[full_table['Province_State'].str.contains(lst, na=False) | full_table['Country_Region'].str.contains(lst, na=False)]
# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']
full_grouped = full_table.groupby(['Date', 'Country_Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

full_grouped
# new cases 

temp = full_grouped.groupby(['Country_Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp.head()
temp = temp.sum().diff().reset_index()

temp
mask = temp['Country_Region'] != temp['Country_Region'].shift(1)

mask
temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan

temp.loc[mask, 'Confirmed']
# renaming columns

temp.columns = ['Country_Region', 'Date', 'New cases', 'New deaths', 'New recovered']

# merging new values

full_grouped = pd.merge(full_grouped, temp, on=['Country_Region', 'Date'])

# filling na with 0

full_grouped = full_grouped.fillna(0)

# fixing data types

cols = ['New cases', 'New deaths', 'New recovered']

full_grouped[cols] = full_grouped[cols].astype('int')

# 

full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)
full_grouped
full_grouped.to_csv('COVID-19-time-series-clean-complete.csv')
import pandas as pd

import altair as alt

full_grouped = pd.read_csv('COVID-19-time-series-clean-complete.csv', parse_dates=['Date'])

sg = full_grouped[full_grouped['Country_Region'] == 'Singapore']
base = alt.Chart(sg).mark_bar().encode(

    x='monthdate(Date):O',

).properties(

    width=500

)
red = alt.value('#f54242')

base.encode(y='Confirmed').properties(title='Total confirmed') | base.encode(y='Deaths', color=red).properties(title='Total deaths')
red = alt.value('#f54242')

base.encode(y='New cases').properties(title='Daily new cases') | base.encode(y='New deaths', color=red).properties(title='Daily new deaths')
import pandas as pd

import altair as alt

full_grouped = pd.read_csv('COVID-19-time-series-clean-complete.csv', parse_dates=['Date'])

countries = ['US', 'Italy', 'China', 'Spain', 'Germany', 'France', 'Iran', 'United Kingdom', 'Switzerland','Singapore']

selected_countries = full_grouped[full_grouped['Country_Region'].isin(countries)]

selected_countries
alt.Chart(selected_countries).mark_circle().encode(

    x='monthdate(Date):O',

    y='Country_Region',

    color='Country_Region',

    size=alt.Size('New cases:Q',

        scale=alt.Scale(range=[0, 1000]),

        legend=alt.Legend(title='Daily new cases')

    ) 

).properties(

    width=1200,

    height=500

)