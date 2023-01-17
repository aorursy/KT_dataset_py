from datetime import datetime



print('Executed at:', datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S (UTC)'))
from kaggle_secrets import UserSecretsClient

import requests



# BE CAREFUL if you fork this kernel and use your own credentials!

user_secrets = UserSecretsClient()

username = user_secrets.get_secret('username')  # DO NOT LEAK

password = user_secrets.get_secret('password')  # DO NOT LEAK



# create a session

session = requests.Session()



# sign in

SIGNIN_URL = 'https://www.kaggle.com/account/email-signin'

session.get(SIGNIN_URL)

data = {

    'email': username,

    'password': password,

    'X-XSRF-TOKEN': session.cookies['XSRF-TOKEN'],

}

session.post(SIGNIN_URL, data=data)



# get the zipped public LB data

PLB_DATA_URL = 'https://www.kaggle.com/c/16531/publicleaderboarddata.zip'

resp = session.get(PLB_DATA_URL)
from zipfile import ZipFile

from io import BytesIO

import pandas as pd



# convert the data to a dataframe

z = ZipFile(BytesIO(resp.content))

df = pd.read_csv(BytesIO(z.read(z.filelist[0].filename)))

df = df.assign(SubmissionDate=pd.to_datetime(df['SubmissionDate']))

df = df.sort_values('Score', ascending=False)

df.head(10)
df.shape
df.info()
# get the best score of each team

best_scores = (

    df.groupby('TeamId')

    .agg({'TeamName': 'last', 'Score': 'max'})

    .reset_index()

    .sort_values('Score', ascending=False)

    .reset_index(drop=True)

    .assign(rank=lambda df: df.index + 1)

)
best_scores.shape
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



thresh = 0.53



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

sns.distplot(best_scores[best_scores['Score'] > thresh]['Score'], ax=ax)

ax.set_title(f'Score (> {thresh}) distribution of the public LB', fontsize=20)

ax.set_xlabel('Score', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=15)
top = best_scores.head(16)  # more than gold

top
top_sbms = (

    pd.merge(top[['TeamId', 'rank']], df, on='TeamId', how='inner')

    .sort_values(['rank', 'SubmissionDate'])

)

top_sbms
import plotly.graph_objects as go

import numpy as np



data = []



for team_id, team in top_sbms.groupby('TeamId', sort=False):

    data.append(go.Scatter(

        x=team['SubmissionDate'],

        y=team['Score'],

        name='{}: {}'.format(team['rank'].iloc[0], team['TeamName'].iloc[0]),

        mode='lines'

    ))



layout = go.Layout({

    'width': 800,

    'height': 600,

    'xaxis': {'title': 'Date'},

    'yaxis': {'title': 'Score', 'range': [0.4, 0.6]},

    'legend': {'orientation': 'h', 'y': 1.7},

})

    

fig = go.Figure(data=data, layout=layout)

fig.show()
layout = go.Layout({

    'width': 800,

    'height': 600,

    'xaxis': {'title': 'Date'},

    'yaxis': {'title': 'Score', 'range': [0.55, 0.6]},

    'legend': {'orientation': 'h', 'y': 1.7},

})

    

fig = go.Figure(data=data, layout=layout)

fig.show()