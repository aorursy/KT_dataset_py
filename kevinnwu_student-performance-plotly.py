import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

%matplotlib inline
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
df.nunique()
fig = make_subplots(rows=3, cols=1, 

                    shared_xaxes=True,

                    subplot_titles=('Writing Score', 'Reading Score', 'Math Score')

                   )



fig.append_trace(go.Histogram(x=df['writing score'], 

                             ), 

                 row=1, col=1)



fig.append_trace(go.Histogram(x=df['reading score'], 

                              #nbinsx = 50,

                             ), 

                 row=2, col=1)



fig.append_trace(go.Histogram(x=df['math score'],

                              #nbinsx = 50,

                             ), 

                 row=3, col=1)



fig.update_layout(height=800, width=800, showlegend=False)

fig.show()
fig = px.histogram(df,

                   x="writing score", 

                   color="gender", 

                   marginal="box",

                   title='Writing Score - Gender', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="reading score", 

                   color="gender", 

                   marginal="box",

                   title='Reading Score - Gender', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="math score", 

                   color="gender", 

                   marginal="box",

                   title='Math Score - Gender', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="writing score", 

                   color="lunch", 

                   marginal="box",

                   title='Writing Score - Lunch', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="reading score", 

                   color="lunch", 

                   marginal="box",

                   title='Reading Score - Lunch', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="math score", 

                   color="lunch", 

                   marginal="box",

                   title='Math Score - Lunch', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="writing score", 

                   color="test preparation course", 

                   marginal="box",

                   title='Writing Score - Test Prepation Course', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="reading score", 

                   color="test preparation course", 

                   marginal="box",

                   title='Reading Score - Test Prepation Course', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
fig = px.histogram(df,

                   x="math score", 

                   color="test preparation course", 

                   marginal="box",

                   title='Math Score - Test Prepation Course', 

                   barmode='overlay',

                  )

fig.update_layout(xaxis=dict(title=''))

fig.show()
data = df

fig = px.box(data,

             x='parental level of education',

             y="writing score", 

             color='parental level of education',

             title='Writing Score - Parental Level Education'

            )



fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=['some high school','high school','some college',"associate's degree", "bachelor's degree", "master's degree"]))

fig.show()
data = df

fig = px.box(data,

             x='parental level of education',

             y="reading score", 

             color='parental level of education',

             title='Reading Score - Parental Level Education'

            )



fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=['some high school','high school','some college',"associate's degree", "bachelor's degree", "master's degree"]))

fig.show()
data = df

fig = px.box(data,

             x='parental level of education',

             y="math score", 

             color='parental level of education',

             title='Math Score - Parental Level Education'

            )



fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=['some high school','high school','some college',"associate's degree", "bachelor's degree", "master's degree"]))

fig.show()
data = df

fig = px.box(data,

             x='race/ethnicity',

             y="writing score", 

             color='race/ethnicity',

             title='Writing Score - Race/Ethnicity'

            )



fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=['group A','group B','group C', 'group D', 'group E']))

fig.show()
data = df

fig = px.box(data,

             x='race/ethnicity',

             y="reading score", 

             color='race/ethnicity',

             title='Reading Score - Race/Ethnicity'

            )



fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=['group A','group B','group C', 'group D', 'group E']))

fig.show()
data = df

fig = px.box(data,

             x='race/ethnicity',

             y="math score", 

             color='race/ethnicity',

             title='Math Score - Race/Ethnicity'

            )



fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=['group A','group B','group C', 'group D', 'group E']))

fig.show()