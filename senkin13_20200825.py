import pandas as pd

import os

import glob

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff



from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
data_df = pd.read_csv("../input/hiranomachi/used-car-price-forecasting-publicleaderboard.csv")

data_df = data_df[data_df['Score']<0.415]

sorted_df = data_df.sort_values(by=['Score'], ascending=False)

sorted_selected_df = sorted_df.drop_duplicates(subset=['TeamId'], keep='first')

first_teams = sorted_selected_df.TeamName
data = []

for team in first_teams:

    dT = data_df[data_df['TeamName'] == team]

    trace = go.Scatter(

        x = dT['SubmissionDate'],y = dT['Score'],

        name=team,

        mode = "markers+lines"

    )

    data.append(trace)



layout = dict(title = 'Public Leaderboard Submissions (current top teams)',

          xaxis = dict(title = 'Submission Date', showticklabels=True), 

          yaxis = dict(title = 'Team Score'),

          #hovermode = 'closest'

         height=800

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='public-leaderboard')
print (os.listdir('../input/hiranomachi/'))
df = pd.DataFrame()

for i in os.listdir('../input/hiranomachi/'):

    if i == 'used-car-price-forecasting-publicleaderboard.csv':

        continue

    sub = pd.read_csv('../input/hiranomachi/'+i)['price']

    df = pd.concat([df,sub],axis=1)

    name = i.split('.')[0]

    df.rename(columns={'price':name},inplace=True)

import seaborn as sns

corr = df.corr()

plt.figure(figsize=(20,20))

sns.heatmap(corr, 

           cmap="Blues", linewidths=2,square=True,

           annot=True, fmt=".2f")

plt.show()