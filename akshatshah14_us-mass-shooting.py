import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import collections



pd.options.display.max_columns = 999



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')
file1 = pd.read_csv("../input/Mass Shootings Dataset.csv", encoding = "ISO-8859-1")
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]



data = [ dict(

        type = 'scattergeo',

        locationmode = 'USA-states',

        lon = file1['Longitude'],

        lat = file1['Latitude'],

        text = file1['Title'],

        mode = 'markers',

        marker = dict(

            size = 8,

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'square',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = scl,

            cmin = 0,

            color = file1['Total victims'],

            cmax = file1['Total victims'].max(),

            colorbar=dict(

                title=""

            )

        ))]



layout = dict(

        title = 'USA shootings area',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.5,

            subunitwidth = 0.5

        ),

    )



fig = dict( data=data, layout=layout )

iplot(fig)
file1.dtypes
file1['Date'] = pd.to_datetime(file1['Date'])
file1.dtypes
file1['year'], file1['month'] = file1['Date'].dt.year, file1['Date'].dt.month
df = pd.DataFrame(file1.groupby('year')['Total victims'].sum())

df['year'] = df.index

df.head()
import matplotlib.pyplot as plt

df.sort_values(by = "Total victims", ascending =False).head(15).plot(x = 'year', y='Total victims', kind = 'bar')
plt.show()
#We see that recently there has been a sudden rise in the mass shootings for all the victims
df1 = pd.DataFrame(file1.groupby('year')['Fatalities'].sum())

df1['year'] = df1.index

df1.head()
import matplotlib.pyplot as plt

df1.sort_values(by = "Fatalities", ascending =False).head(15).plot(x = 'year', y='Fatalities', kind = 'bar')

plt.show()

file1['Mental Health Issues'] = file1['Mental Health Issues'].astype(str).str.lower()

file1['Mental Health Issues'].value_counts().plot(kind='bar')

plt.xlabel('Mental Health Issue')

plt.ylabel('Count')

plt.show()
df1=file1['Mental Health Issues'].value_counts()

df1
file1['Race'] = file1['Race'].astype(str).str.lower()

file1['Race'].value_counts().plot(kind='bar')

plt.xlabel('Race')

plt.ylabel('Count')

plt.show()
file1['Gender'] = file1['Gender'].astype(str).str.lower()

file1['Gender'].replace(['M','F','m/f','male/female'],['Male','Female','unknown','unknown'] ,inplace=True)

file1['Gender'].value_counts().plot(kind='bar')

plt.xlabel('Gender')

plt.ylabel('Count')

plt.show()
df2 = pd.DataFrame(file1.groupby('year')['Injured'].sum())
df2['year'] = df2.index
df2.head()
df2.sort_values(by = "Injured", ascending =False).head(15).plot(x = 'year', y='Injured', kind = 'bar')

plt.show()
from ipywidgets import interact 

def func(No_of_bins):

    df3 = pd.DataFrame(file1.groupby('year')['Injured'].sum())

    df3['year'] = df2.index

    df3.head()

    df3.sort_values(by = "Injured", ascending =False).head(No_of_bins).plot(x = 'year', y='Injured', kind = 'bar')

    plt.show()

#     x = options

#     return x

interact(func, No_of_bins = [10, 11, 12])



# InjuredVsYear(x)

def InjuredVsYear(t):

    df3 = pd.DataFrame(file1.groupby('year')['Injured'].sum())

    df3['year'] = df2.index

    df3.head()

    df3.sort_values(by = "Injured", ascending =False).head(t).plot(x = 'year', y='Injured', kind = 'bar')

    plt.show()
labels = ['male','unknown','female']

values = [289,26,5]



trace = go.Pie(labels=labels, values=values)

iplot([trace])