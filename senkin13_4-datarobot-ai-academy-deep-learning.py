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



print (os.listdir('../input/4th-datarobot-aiacademy/'))
data_df = pd.read_csv("../input/4th-datarobot-aiacademy/4th-datarobot-ai-academy-deep-learning-publicleaderboard.csv")

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



layout = dict(title = 'Leaderboard Submissions',

          xaxis = dict(title = 'Submission Date', showticklabels=True), 

          yaxis = dict(title = 'Team Score'),

          #hovermode = 'closest'

         height=800

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='Leaderboard')
df = pd.DataFrame()

for i in os.listdir('../input/4th-datarobot-aiacademy/'):

    if i == '4th-datarobot-ai-academy-deep-learning-publicleaderboard.csv':

        continue

    sub = pd.read_csv('../input/4th-datarobot-aiacademy/'+i)['price']

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
YasuNiina = pd.read_csv('../input/4th-datarobot-aiacademy/YasuNiina.csv')

masayuki_ishida_sb = pd.read_csv('../input/4th-datarobot-aiacademy/masayuki_ishida_sb.csv')

minami_ds = pd.read_csv('../input/4th-datarobot-aiacademy/minami_ds.csv')

ensemble = YasuNiina.copy()

ensemble['YasuNiina'] = ensemble['price']

ensemble['masayuki_ishida_sb'] = masayuki_ishida_sb['price']

ensemble['minami_ds'] = minami_ds['price']

ensemble['price'] = ensemble['YasuNiina']*0.5 + ensemble['masayuki_ishida_sb']*0.3 + ensemble['minami_ds']*0.2

ensemble[['id','price']].to_csv('submission.csv',index=False)