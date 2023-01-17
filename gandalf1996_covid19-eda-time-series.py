!pip install chart-studio
from IPython.display import Image

import chart_studio.plotly as py



Image("../input/covid-19-image/COVID19.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import chart_studio.plotly as py

from plotly.graph_objs import *

from IPython.display import Image

pd.set_option('display.max_rows', None)



import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# For Density plots

from plotly.tools import FigureFactory as FF



import datetime

import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
covid_19_data.head()
time_series_covid_19_confirmed.head()
time_series_covid_19_deaths.head()
time_series_covid_19_recovered.head()
def group_covid(data,country):

    cases = data.groupby(country).size()

    cases = np.log(cases)

    cases = cases.sort_values()

    

    # Visualize the results

    fig=plt.figure(figsize=(35,7))

    plt.yticks(fontsize=8)

    cases.plot(kind='bar',fontsize=12,color='orange')

    plt.xlabel('')

    plt.ylabel('Number of cases',fontsize=10)



group_covid(covid_19_data,'Country/Region')
cases = covid_19_data.groupby(['Country/Region']).size()

cases = cases.sort_values()



grouped_df = pd.DataFrame(cases)



grouped_df['Count'] = pd.Series(cases).values

grouped_df['Country/Region'] = grouped_df.index

grouped_df['Log Count'] = np.log(grouped_df['Count'])

grouped_df.head()
fig = go.Figure(go.Scatter(

    x = grouped_df['Country/Region'],

    y = grouped_df['Count'],

    text=['Line Chart with lines and markers'],

    name='Country/Region',

    mode='lines+markers',

    marker_color='#56B870'

))



fig.update_layout(

    height=800,

    title_text='COVID-19 cases across nations using line chart',

    showlegend=True

)



fig.show()
fig = go.Figure(go.Scatter(

    x = grouped_df['Country/Region'],

    y = grouped_df['Log Count'],

    text=['Line Chart with lines and markers'],

    name='Country/Region',

    mode='lines+markers',

    marker_color='#56B870'

))



fig.update_layout(

    height=800,

    title_text='COVID-19 cases across nations using line chart (Log scale)',

    showlegend=True

)



fig.show()
fig = go.Figure(go.Bar(

    x = grouped_df['Country/Region'],

    y = grouped_df['Log Count'],

    text=['Bar Chart'],

    name='Countries',

    marker_color=grouped_df['Count']

))



fig.update_layout(

    height=800,

    title_text='COVID-19 cases across nations using bar chart (Log scale)',

    showlegend=True

)



fig.show()
fig = go.Figure(go.Bar(

    x = grouped_df['Country/Region'],

    y = grouped_df['Log Count'],

    text=['Bar Chart'],

    name='Countries',

    marker_color=grouped_df['Log Count']

))



fig.update_layout(

    height=800,

    title_text='COVID-19 cases across nations using bar chart (Log scale)',

    showlegend=True

)



fig.show()
grouped_df.head()
covid_countries = covid_19_data.groupby('Country/Region')['ObservationDate'].value_counts().reset_index(name='t')

covid_countries['Count'] = covid_19_data.groupby('Country/Region')['ObservationDate'].transform('size')



covid_countries.head()
# Create traces

fig = go.Figure()



fig.add_trace(go.Scatter(

    x = covid_countries['ObservationDate'],

    y = covid_countries['Country/Region'],

    text=['Line Chart with lines and markers'],

    name='Countries',

    mode='markers',

    

))



fig.update_layout(

    height=800,

    title_text='COVID-19 case occurences across nations',

    showlegend=True

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x = covid_countries['ObservationDate'],

                y = covid_countries['Country/Region'],

                name='Country/Region',

                marker_color=covid_countries['Count']

                ))



fig.update_layout(

    height=800,

    title_text='COVID-19 cases visualized over time across nations',

    

    showlegend=True

)



fig.show()


confirmed_df = covid_19_data.groupby('Confirmed')['Country/Region'].value_counts().reset_index(name='Conf')

confirmed_df['Conf'] = covid_19_data.groupby('Confirmed')['Country/Region'].transform('size')



confirmed_df.head()
fig = px.pie(confirmed_df, values='Conf', names='Country/Region')

fig.show()
deaths_df = covid_19_data.groupby('Deaths')['Country/Region'].value_counts().reset_index(name='Death')

deaths_df['Death'] = covid_19_data.groupby('Deaths')['Country/Region'].transform('size')



deaths_df.head()
fig = px.pie(deaths_df, values='Death', names='Country/Region')

fig.show()
recover_df = covid_19_data.groupby('Recovered')['Country/Region'].value_counts().reset_index(name='Recover')

recover_df['Recover'] = covid_19_data.groupby('Recovered')['Country/Region'].transform('size')



recover_df.head()
fig = px.pie(recover_df, values='Recover', names='Country/Region')

fig.show()
sns.color_palette("cubehelix", 8)

sns.set_style("whitegrid", {'axes.grid' : False})



sns.color_palette("cubehelix", 8)

sns.distplot(confirmed_df['Conf'],bins=100,hist=False,   label="Confirmed Cases");

# sns.distplot(deaths_df['Death'],bins=100,hist=False,   label="Deaths");

# sns.distplot(recover_df['Recover'],bins=100,hist=False,   label="Recovered Cases");





plt.legend();