!pip install ipython-autotime

%matplotlib inline

%load_ext autotime
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as pty

import plotly.express as px

import plotly.graph_objects as go
data_path = '../input/cc-live-proj/'

train = pd.read_csv(data_path + 'AI-DataTrain.csv')

train = train.drop(columns=['Unnamed: 0'])

test = pd.read_csv(data_path + 'AI-DataTest.csv')
train.head()
test.head()
train.describe()
test.describe()
print('Shape of train data:',train.shape)

print('Shape of test data:',test.shape)
counts_train = train.apply(pd.Series.value_counts)

counts_train
questions_train = counts_train.columns.values



fig = go.Figure(data=[

    go.Bar(name='Incorrect', x=questions_train, y=counts_train.loc[0].values),

    go.Bar(name='Correct', x=questions_train, y=counts_train.loc[1].values)

])

# Change the bar mode

fig.update_layout(

    autosize=True,

    barmode='stack',

    title_text='Correct and incorrect attempts of Questions in train data',

    xaxis_tickangle=-45,

    xaxis_title="Question Number",

    yaxis_title="Number of times attempted correct and incorrect",

    title=dict(

        y=0.9,

        x=0.5,

        xanchor='center',

        yanchor='top',

        font=dict(

            size=24,

        )

    )

)

fig.show()
counts_test = test.apply(pd.Series.value_counts)

counts_test
questions_test = counts_test.columns.values



fig = go.Figure(data=[

    go.Bar(name='Incorrect', x=questions_test, y=counts_test.loc[0].values),

    go.Bar(name='Correct', x=questions_test, y=counts_test.loc[1].values)

])

# Change the bar mode

fig.update_layout(

    barmode='stack',

    title_text='Correct and incorrect attempts of Questions in Test data',

    xaxis_tickangle=-45,

    xaxis_title="Question Number",

    yaxis_title="No. of times attempted correct and incorrect",

    title=dict(

        y=0.9,

        x=0.5,

        xanchor='center',

        yanchor='top',

        font=dict(

            size=24,

        )

    )

)

fig.show()
train_copy = train.copy(deep=True)

train_copy.loc[:,'Score']=train_copy.sum(axis=1)

train_copy.head()
fig = px.histogram(train_copy,

                   x='Score',

                   histnorm='probability density',

                   opacity=0.6,

                   color_discrete_sequence=['green','blue']

                  )



fig.update_layout(

    title=dict(

        text='Scores Histigram (Train set)',

        y=0.95,

        x=0.5,

        xanchor='center',

        yanchor='top',

        font=dict(

            size=24,

        )

    )

)

fig.show()
test_copy = test.copy(deep=True)

test_copy.loc[:,'Score']=test_copy.sum(axis=1)

test_copy.head()
fig = px.histogram(test_copy,

                   x='Score',

                   histnorm='probability density',

                   opacity=0.6,

                   color_discrete_sequence=['green','blue']

                  )



fig.update_layout(

    title=dict(

        text='Scores Histigram (Test set)',

        y=0.95,

        x=0.5,

        xanchor='center',

        yanchor='top',

        font=dict(

            size=24,

        )

    )

)

fig.show()