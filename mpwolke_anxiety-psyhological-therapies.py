# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsanxietycsv/anxiety.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'anxiety.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sns.countplot(df["Final"])

plt.xticks(rotation=90)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://ars.els-cdn.com/content/image/1-s2.0-S0165178120306521-gr2.jpg',width=400,height=400)
df["Final"].plot.hist()

plt.show()
fig = px.bar(df, 

             x='GAD', y='Avg wait for therapy (days)', color_discrete_sequence=['#27F1E7'],

             title='Anxiety - Avg wait for therapy', text='GAD')

fig.show()
fig = px.bar(df, 

             x='GAD', y='therapist_agreeing_that_cpd_is_sufficient', color_discrete_sequence=['#f5ef42'],

             title='Anxiety - Therapist agreeing that CPD is sufficient', text='GAD')

fig.show()
fig = px.bar(df, 

             x='GAD', y='Female', color_discrete_sequence=['#f56642'],

             title='Anxiety - Females', text='GAD')

fig.show()
fig = px.bar(df, 

             x='GAD', y='Male', color_discrete_sequence=['#424ef5'],

             title='Anxiety - Males', text='GAD')

fig.show()
fig = px.density_contour(df, x="GAD", y="Final", color_discrete_sequence=['purple'])

fig.show()
fig = px.scatter(df.dropna(), x='Expected',y='Final', trendline="GAD", color_discrete_sequence=['purple'])

fig.show()
fig = px.line(df, x="Unemployed_seeking work", y="GAD", color_discrete_sequence=['green'], 

              title="Unemployed seeking work Anxiety")

fig.show()
px.histogram(df, x='Final', color='GAD')
fig = px.bar(df, x= "OCD_problem", y= "Female")

fig.show()
cnt_srs = df['PanicDisorder_problem'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'viridis',

        reversescale = True

    ),

)



layout = dict(

    title='Panic Disorder Problem',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="PanicDisorder_problem")
cnt_srs = df['Depression_problem'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'BuPu',

        reversescale = True

    ),

)



layout = dict(

    title='Depression Problem',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Depression_problem")
cnt_srs = df['Socialphob_problem'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'rainbow',

        reversescale = True

    ),

)



layout = dict(

    title='Socialphobia Problem',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Socialphob_problem")
dfgr = df.groupby('Final').count()['Expected'].reset_index().sort_values(by='Expected',ascending=False)

dfgr.style.background_gradient(cmap='PuOr')
fig = go.Figure(go.Funnelarea(

    text =dfgr.Final,

    values = dfgr.Expected,

    title = {"position": "top center", "text": "Funnel-Chart of Anxiety Distribution"}

    ))

fig.show()
fig = go.Figure(data=[go.Bar(

            x=dfgr['Expected'][0:10], y=dfgr['Final'][0:10],

            text=dfgr['Final'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Anxiety Therapies',

    xaxis_title="Expected",

    yaxis_title="Final",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=dfgr['Expected'][0:10],

    y=dfgr['Final'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Anxiety Therapies',

    xaxis_title="Expected",

    yaxis_title="Final",

)

fig.show()