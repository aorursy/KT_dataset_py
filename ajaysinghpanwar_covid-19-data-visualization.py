import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
covid_df = pd.read_csv("/kaggle/input/corona-virus-report/country_wise_latest.csv")
covid_df.head()
covid_df.isnull().sum()
import plotly.graph_objects as go



fig = go.Figure(data = go.Choropleth(

    locations = covid_df['Country/Region'],

    locationmode = 'country names',

    z = covid_df['Confirmed'],

    text = covid_df['Country/Region'],

    colorscale = 'viridis',

    autocolorscale = False,

    reversescale = True,

    marker_line_color = 'darkgray',

    marker_line_width = 0.5,

    colorbar_title = 'Confirmed Covid Cases',

))



fig.update_layout(

    title_text = 'Covid-19 Data',

    geo = dict(

    showframe = False,

    showcoastlines = False,

    projection_type = 'orthographic'

    ))



fig.show()
top_15_df = covid_df.nlargest(15, 'Confirmed')
fig = px.bar(top_15_df, x = 'Country/Region', y = 'Confirmed',

             hover_data = ['Deaths','Recovered'], color = 'Active',

             title = 'Top 15 countries with most no. of Covid-19 cases')



fig.show()
top_15_death = covid_df.nlargest(15, 'Deaths / 100 Cases')
fig = px.pie(top_15_death , values = 'Deaths / 100 Cases', 

             names='Country/Region', title ='Death Rate Per 100 Cases')

fig.show()
top_15_inc = covid_df.nlargest(15, '1 week % increase')
fig = px.pie(top_15_inc , values = '1 week % increase', 

             names='Country/Region', title ='% Increase of cases in a week')

fig.show()