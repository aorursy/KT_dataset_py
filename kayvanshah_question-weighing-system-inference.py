!pip install ipython-autotime

%load_ext autotime
import os,re



import numpy as np

import pandas as pd



import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt



import tensorflow as tf
data_path = '../input/cc-live-proj/'

data = pd.read_csv(data_path + 'AI-DataTrain.csv')

test_data = pd.read_csv(data_path + 'AI-DataTest.csv')



print('Data shape: ',data.shape)

print('Test_data shape:', test_data.shape)
test_data.head()
model = tf.keras.models.load_model('../input/question-weighing-system-training/qws.h5')

model.summary()
model.save('model')

x = tf.keras.models.load_model('/kaggle/working/model')

x.summary()
def get_data_features(df):

    pdata = df.apply(pd.Series.value_counts)

    pdata = pdata.T[[0,1]].values

    pdata = pdata/df.shape[0]

    return pdata





def estimate_weights(features):

    weights = model.predict(features)

    return weights





def output_csv(df,weights):

    questions_pool = df.columns.values

    dataqw = {'Question':questions_pool, 'Weights':weights}

    qwdf = pd.DataFrame(data=dataqw)

    return qwdf
def get_weights(df):

    features = get_data_features(df)

    weights = estimate_weights(features).T[0]

    csv = output_csv(df,weights)

    return csv
df = get_weights(test_data)

df.to_csv('submission.csv')

df
fig = go.Figure(data=[

    go.Bar(name='Weights', x=df.Question, y=df.Weights)

])

# Change the bar mode

fig.update_layout(

    title_text='Weights for Question in Test data',

    xaxis_tickangle=-45,

    xaxis_title="Question Number",

    yaxis_title="Weights",

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