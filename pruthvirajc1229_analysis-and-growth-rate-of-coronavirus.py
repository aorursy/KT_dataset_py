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
import numpy as np 

import pandas as pd 



import ipywidgets as widgets

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from IPython.display import clear_output

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data.head()
data.info()
data['Last Update'] = data['Last Update'].apply(pd.to_datetime)

#data.drop(['Sno'],axis=1,inplace=True)

data.head()
countries = data['Country'].unique().tolist()

print(countries)



print("\nTotal countries affected by virus: ",len(countries))
from datetime import date

data_new = data[data['Last Update'] > pd.Timestamp(date(2020,1,30))]



data_new.head()
cases = pd.DataFrame(data_new.groupby('Country')['Confirmed'].sum())

cases['Country'] = cases.index

cases.index=np.arange(1,31)



global_cases = cases[['Country','Confirmed']]

#global_cases.sort_values(by=['Confirmed'],ascending=False)

global_cases
from datetime import date

data1 = data[data['Last Update'] > pd.Timestamp(date(2020,2,3))]



data1.info()
cases = pd.DataFrame(data1.groupby('Country')['Confirmed'].sum())

cases['Country'] = cases.index

cases.index=np.arange(1,30)



global_cases = cases[['Country','Confirmed']]

global_cases.sort_values(by=['Confirmed'],ascending=False)

global_cases
# A look at the different cases - confirmed, death and recovered



print('Globally Confirmed Cases: ',data_new['Confirmed'].sum())

print('Global Deaths: ',data_new['Deaths'].sum())

print('Globally Recovered Cases: ',data_new['Recovered'].sum())
print('Globally Confirmed Cases: ',data1['Confirmed'].sum())

print('Global Deaths: ',data1['Deaths'].sum())

print('Globally Recovered Cases: ',data1['Recovered'].sum())
# Let's look the various Provinces/States affected



data_new.groupby(['Country','Province/State']).sum()
#data1.groupby(['Country','Province/State']).sum()
# Provinces where deaths have taken place



data_new.groupby('Country')['Deaths'].sum().sort_values(ascending=False)[:5]
# Lets also look at the Recovered stats

data_new.groupby('Country')['Recovered'].sum().sort_values(ascending=False)[:5]
## Cases of infection according to date

Confirmed_count_date=pd.DataFrame(data.groupby(['Last Update'])['Confirmed'].sum())

labels=Confirmed_count_date.index

sizes=Confirmed_count_date['Confirmed']

explode = None   # explode 1st slice



print(Confirmed_count_date)
#Mainland China

China = data_new[data_new['Country']=='Mainland China']

China
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=China[1:],

            label="Confirmed", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=China[1:],

            label="Recovered", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 30), ylabel="",

       xlabel="Patients")

sns.despine(left=True, bottom=True)
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=data[1:],

            label="Confirmed", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=data[1:],

            label="Recovered", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 30), ylabel="",

       xlabel="Patients")

sns.despine(left=True, bottom=True)
def plot_trend(df, y_feature, loc_feature='Country', loc='All', freq='Daily'):

    df = df.copy()

    # filter by Location

    if loc == 'All':

        df = df.sort_values('Last Update').reset_index(drop=True)

    else:

        df = df[df[loc_feature] == loc].sort_values('Last Update').reset_index(drop=True)

    # group by freq

    df.index = df['Last Update']

    if freq == 'Daily':

        df = df.groupby(pd.Grouper(freq='D'))[y_feature].sum().reset_index()

    elif freq == 'Hourly':

        df = df.groupby(pd.Grouper(freq='H'))[y_feature].sum().reset_index()

        

    trace1 = go.Scatter(

                        x = df['Last Update'],

                        y = df[y_feature],

                        mode = "lines",

                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'))



    traces = [trace1]

    layout = dict(title = f'[{y_feature}] {freq}',

                  xaxis= dict(title='Date',ticklen= 5,zeroline= False)

                 )

    fig = dict(data=traces, layout = layout)

    iplot(fig)



# UI

country_dropdown_1 = widgets.Dropdown(

    options=data['Country'].unique().tolist() + ['All'],

    value='All',

    description='Country:',

    disabled=False,

)



y_dropdown_1 = widgets.Dropdown(

    options=['Confirmed', 'Deaths', 'Recovered'],

    value='Confirmed',

    description='Stats:',

    disabled=False,

)



def on_value_change(change):

    clear_output()

    display(country_dropdown_1, y_dropdown_1)

    plot_trend(data, y_feature=y_dropdown_1.value, loc=country_dropdown_1.value, freq='Daily')



country_dropdown_1.observe(on_value_change, names='value')

y_dropdown_1.observe(on_value_change, names='value')

# trigger init

on_value_change(None)

data['Last Update'] = pd.to_datetime(data['Last Update'])

data['Day'] = data['Last Update'].apply(lambda x : x.day)

data['Hour'] = data['Last Update'].apply(lambda x : x.hour)



#data.drop(['Sno'], axis=1 , inplace=True)

data = data[data['Confirmed'] != 0]

data
data.Date.unique()
data.Country.unique()
countries=data.groupby(["Country"]).sum().reset_index()

countries
data.shape
#import plotly.graph_objects as go



days = list(data.Day.unique())



fig = go.Figure()

fig.add_trace(go.Bar(

    x=days,

    y=list(data.groupby('Day')['Confirmed'].sum()),

    name='Confirmed per day',

    marker_color='blue'

))

fig.add_trace(go.Bar(

    x=days,

    y=list(data.groupby('Day')['Deaths'].sum()),

    name='Deaths per day',

    marker_color='red'

))



fig.add_trace(go.Bar(

    x=days,

    y=list(data.groupby('Day')['Recovered','Confirmed','Deaths'].sum()),

    name='Recovered per day',

    marker_color='green'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()