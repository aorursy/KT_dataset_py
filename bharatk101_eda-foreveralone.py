# import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as plty
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

%matplotlib inline
# load dataset
df = pd.read_csv('../input/foreveralone.csv')
df.head()
df.info()
df.describe()
df.duplicated().any()
df.isna().any()
df.job_title.value_counts()
# change dataype to int
df['friends'] = df['friends'].astype(np.int64)
df.info()
#drop rows with null values
df.dropna(inplace=True)
df.isna().any()
# strip stings with white space
df['job_title'] = df.job_title.str.strip()
# Function to replace job_title values
def replace_text(what, to):
    df.replace(what, to, inplace= True)
replace_text('student', 'Student')
replace_text('none', 'None')
replace_text("N/a", 'None')
replace_text('na', 'None')
replace_text('-', 'None')
replace_text('.', 'None')
replace_text('*', 'None')
replace_text('ggg', 'None')
df.job_title.value_counts()
df.gender.value_counts()
# Gender counts

data = [go.Bar(x = ['Male', 'Female', 'Transgender Male', 'Transgender Female'],
              y = df.gender.value_counts())]
layout = go.Layout(
    title='Gender Frequency',
    xaxis=dict(
        title='Gender'
    ),
    yaxis=dict(
        title='Count'
        )
    )

fig = go.Figure(data=data, layout=layout)
plty.offline.iplot(fig)
# sexuality freqency

df.sexuallity.value_counts()
# Sexuality counts

data = [go.Bar(x = ['Straight', 'Bisexual', 'Gay/Lesbian'],
              y = df.sexuallity.value_counts())]
layout = go.Layout(
    title='Sexuality Frequency',
    xaxis=dict(
        title='Sexuality'
    ),
    yaxis=dict(
        title='Count'
        )
    )

fig = go.Figure(data=data, layout=layout)
plty.offline.iplot(fig)
# body weight

df.bodyweight.value_counts()
def univariate_bar(column, ttitle, xlabel, ylabel):
    temp = pd.DataFrame({column:df[column].value_counts()})
    df1 = temp[temp.index != 'Unspecified']
    df1 = df1.sort_values(by=column, ascending=False)
    data  = go.Data([
                go.Bar(
                  x = df1.index,
                  y = df1[column],
            )])
    layout = go.Layout(
            title = ttitle,
        xaxis=dict(
            title=xlabel
        ),
        yaxis=dict(
            title=ylabel
            )
    )
    fig  = go.Figure(data=data, layout=layout)
    return plty.offline.iplot(fig)
univariate_bar('bodyweight', 'Bodyweight Frequency', 'Weight', 'Counts')
univariate_bar('depressed', 'Number of People Depressed', ' ', 'Count')
univariate_bar('social_fear', 'Number of People having Social Fear', ' ', 'Count')
univariate_bar('attempt_suicide', 'Number of people attempted suicide', ' ', 'Count')
age = df['age']

trace = go.Histogram(x = age)

data = [trace]

layout = go.Layout(
    title = 'Age Distribution',
    xaxis = dict(
        title = 'Age'
    ),
    yaxis = dict(
        title ='Count'
    ))

fig = go.Figure(data, layout)
plty.offline.iplot(fig)
# Distribution of Friends

friends = df['friends']

trace = go.Histogram(x = friends)
data = [trace]

layout = go.Layout(
    title = 'Friends Distribution',
    xaxis = dict(
    title = 'Friend Count'),
    yaxis = dict(
    title = 'Count')
    )

fig = go.Figure(data, layout)
plty.offline.iplot(fig)
male = df[df['gender'] == 'Male' ]
female = df[df['gender'] == 'Female' ]

male_age = male['age']
female_age = female['age']
trace1 = go.Histogram(x = male_age, 
                      name = 'Male',
                     opacity = 0.5)
trace2 = go.Histogram(x = female_age,
                      name = 'Female',
                     opacity = 0.5)

data = [trace1, trace2]

layout = go.Layout(
    title = 'Age Distribution on Gender',
    barmode='overlay',
    xaxis = dict(
    title = 'Age'),
    yaxis = dict(
    title = 'Count')
    )

fig = go.Figure(data, layout)
plty.offline.iplot(fig)
male_friends = male['friends']
female_friends = female['friends']
trace1 = go.Histogram(x = male_friends, 
                      name = 'Male',
                     opacity = 0.5)
trace2 = go.Histogram(x = female_friends,
                      name = 'Female',
                     opacity = 0.5)

data = [trace1, trace2]

layout = go.Layout(
    title = 'Friends Distribution on Gender',
    barmode='overlay',
    xaxis = dict(
    title = 'Friends'),
    yaxis = dict(
    title = 'Count')
    )

fig = go.Figure(data, layout)
plty.offline.iplot(fig)