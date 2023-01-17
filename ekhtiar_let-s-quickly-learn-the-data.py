# our imports

import numpy as np

import pandas as pd

import plotly.express as px
# loading all the data into dataframe

specs_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

train_labels_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

train_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
specs_df.head()
train_df.head()
test_df.head()
train_labels_df.head()
# count activity per installation_id

count_per_installation_id = train_df.groupby(['installation_id']).count()['event_id']

# use plotly express to draw a scatterplot of activity per installation_id

fig = px.scatter(x=count_per_installation_id.index, y=count_per_installation_id.values,

                title='Total Events Per Installation')

fig.show()
session_per_installation_id = train_df.groupby(['installation_id']).game_session.nunique()

fig = px.scatter(x=session_per_installation_id.index, y=session_per_installation_id.values,

                title='Sessions Per Installation')

fig.show()