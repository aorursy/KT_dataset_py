import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from wordcloud import WordCloud



init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/data-science-bowl-2019/train.csv", parse_dates=["timestamp"])

df_train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

df_specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

df_sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

df_test = pd.read_csv("../input/data-science-bowl-2019/test.csv", parse_dates=["timestamp"])
df_train.info()
df_test.info()
df_train.head()
print(df_train.loc[:, df_train.isnull().any()].isnull().sum())
train_types = df_train["type"].value_counts()

test_types = df_test["type"].value_counts()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(

    go.Pie(values=train_types, labels=train_types.index.tolist(), name="Train" , hole=.3),

    1, 1)



fig.add_trace(

    go.Pie(values=test_types, labels=test_types.index.tolist(), name="Test" , hole=.3),

    1, 2)



fig.update_traces(hoverinfo='label+percent+value', textinfo='percent', textfont_size=17, textposition="inside",

                  marker=dict(colors=['gold', 'mediumturquoise', 'darkorange', 'plum'],  

                              line=dict(color='#000000', width=2)))



fig.update_layout(

    title_text="Media Type of The Game or Video",

    height=500, width=800,

    annotations=[dict(text='Train', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Test', x=0.81, y=0.5, font_size=20, showarrow=False)])



fig.show()
train_worlds = df_train["world"].value_counts()

test_worlds = df_test["world"].value_counts()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]])



fig.add_trace(

    go.Bar(y=train_worlds.values, x=train_worlds.index),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=test_worlds.values, x=test_worlds.index),

    row=1, col=2)



fig.update_layout(

    title_text="World of Apps",

    height=500, width=800, showlegend=False)



fig['layout']['xaxis1'].update(title='Train')

fig['layout']['xaxis2'].update(title='Test')



fig.show()
eventbyinstallation = df_train.groupby(["installation_id"])["event_code"].nunique()



fig = px.histogram(x=eventbyinstallation,

                   title='Unique Event Code Count by Installation Id',

                   opacity=0.8,

                   color_discrete_sequence=['indianred'])



fig.update_layout(

    yaxis_title_text='',

    xaxis_title_text='',

    height=500, width=800)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.8

                 )



fig.show()
