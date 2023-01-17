# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



from matplotlib import animation, rc

from IPython.display import HTML, Image

rc('animation', html='html5')



!pip install bar_chart_race

import bar_chart_race as bcr





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

df1.head()
df = df1.groupby(["date"])["total_cases"].sum().reset_index().sort_values("total_cases",ascending=False).reset_index(drop=True)

df
# Converting Date format

df['date'] = pd.to_datetime(df['date'])

df['date'].dtype

# Year wise data

# mean price 

df_year = df.groupby(df.date.dt.year)['total_cases'].agg('mean').reset_index()

df_year.head()
# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots(figsize=(8,6))

ax.set_xlim((2019, 2020))

ax.set_ylim(np.min(df_year.total_cases), np.max(df_year.total_cases)+1)

ax.set_xlabel('Year',fontsize = 14)

ax.set_ylabel('total_cases',fontsize = 14)

ax.set_title('Total Cases over the Years',fontsize = 18)

ax.xaxis.grid()

ax.yaxis.grid()

ax.set_facecolor('#000000') 

line, = ax.plot([], [], lw=4,color='green')



# initialization function: plot the background of each frame

def init():

    line.set_data([], [])

    return (line,)





# animation function. This is called sequentially

def animate(i):

    d = df_year.iloc[:int(i+1)] #select data range

    x = d.date

    y = d.total_cases

    line.set_data(x, y)

    return (line,)



anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=40, repeat=True)
anim
# Week wise data 2020 Jan to April

mask = (df['date'] > '2019-12-31') & (df['date'] <= '2020-06-21')

data_2020 = df[mask]

# mean price 

data_2020_weekly = data_2020.set_index('date').resample('W').mean().reset_index()

data_2020_weekly.head()
import datetime

fig, ax = plt.subplots(figsize=(8,6))



ax.set_xlim([datetime.date(2020, 1, 2), datetime.date(2020, 3, 31)])

ax.set_ylim(np.min(data_2020_weekly.total_cases), np.max(data_2020_weekly.total_cases)+1)

ax.set_xlabel('date',fontsize = 14)

ax.set_ylabel('total_cases',fontsize = 14)

ax.set_title('Total Cases Per Week 2020 Jan - Jun',fontsize = 18)

ax.xaxis.grid()

ax.yaxis.grid()

ax.set_facecolor('#000000') 

line, = ax.plot([], [], lw=4,color='green')



# initialization function: plot the background of each frame

def init():

    line.set_data([], [])

    return (line,)





# animation function. This is called sequentially

def animate(i):

    d = data_2020_weekly.iloc[:int(i+1)] #select data range

    x = d.date

    y = d.total_cases

    line.set_data(x, y)

    return (line,)



anim = animation.FuncAnimation(fig, animate, init_func=init,

                               frames=14, repeat=True)
anim
# Lets take only few countries

cols = ['date','total_cases','cvd_death_rate']

data_deaths = df1[cols]

data_deaths.set_index("date", inplace = True) 

data_deaths.head()
#bcr.bar_chart_race(df=data_deaths, filename=None, figsize = (3.5,3),title='COVID-19 Deaths')
from IPython.display import Image

sns.set(style="darkgrid", palette="pastel", color_codes=True)

sns.set_context("paper")

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "seaborn"

from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode, iplot,plot

init_notebook_mode(connected=True)
# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])



# Add traces

fig.add_trace(

    go.Scatter(x=df1.date, y=df1.cvd_death_rate, name="cvd_death_rate"),

    secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=df1.date, y=df1.cvd_death_rate, name="Covid19 Death Rate",line = dict(color = 'orangered')),

    secondary_y=True,

)



# Add figure title

fig.update_layout(

#     title_text="Total cases vs Price"

    title='<b>Total cases by Covid19</b>',

    plot_bgcolor='linen',

#     paper_bgcolor = 'grey',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=2,

                     label='2m',

                     step='month',

                     stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible = True

        ),

        type='date'

    )

)



# Set x-axis title

fig.update_xaxes(title_text="<b>date</b>")



# Set y-axes titles

fig.update_yaxes(title_text="<b>total_cases</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>cvd_death_rate</b>", secondary_y=True)



iplot(fig)
# Impact till of Jun 21, 2020.

cols = ['total_cases','date','cvd_death_rate','gdp_per_capita','extreme_poverty',

        'life_expectancy','diabetes_prevalence']



cordata = pd.DataFrame(df1[cols].corr(method ='pearson'))



fig = go.Figure(data=go.Heatmap(z=cordata,x=cols,y=cols,colorscale='burgyl'))



iplot(fig)
cordata