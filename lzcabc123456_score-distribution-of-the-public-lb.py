from datetime import datetime



print('Executed at:', datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S (UTC)'))
from kaggle_secrets import UserSecretsClient

import requests



# get the zipped public LB data

PLB_DATA_URL = 'https://www.kaggle.com/c/18599/publicleaderboarddata.zip'

# resp = session.get(PLB_DATA_URL)
from zipfile import ZipFile

# from io import BytesIO

import pandas as pd



# convert the data to a dataframe<div class="ListItemHeader_NameText-sc-1pyomlz cEwhBG">m5-forecasting-accuracy-publicleaderboard.xls</div>

# z = ZipFile(BytesIO(resp.content))

df = pd.read_csv("../input/lbdata/m5-forecasting-accuracy-publicleaderboard.xls")

df = df.assign(SubmissionDate=pd.to_datetime(df['SubmissionDate']))

df = df.sort_values('Score', ascending=False)

df.head(10)
df.shape
df.info()
# get the best score of each team

best_scores = (

    df.groupby('TeamId')

    .agg({'TeamName': 'last', 'Score': 'min'})

    .reset_index()

    .sort_values('Score', ascending=False)

    .reset_index(drop=True)

    .assign(rank=lambda df: -df.index +3771)

)
best_scores.shape
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



thresh = 0.53



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

sns.distplot(best_scores[best_scores['Score'] < thresh]['Score'], ax=ax)

ax.set_title(f'Score (< {thresh}) distribution of the public LB', fontsize=20)

ax.set_xlabel('Score', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=15)
top = best_scores.tail(20)  # more than gold

top
top_sbms = (

    pd.merge(top[['TeamId', 'rank']], df, on='TeamId', how='inner')

    .sort_values(['rank', 'SubmissionDate'])

)

top_sbms[top_sbms['TeamName']=='BUAA_Stat']
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize 

figsize(18, 8)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']



valve = 0.48



for team_id, team in top_sbms.groupby('TeamId', sort=False):

    x=team[team["Score"]<valve]['SubmissionDate']

    y=team[team["Score"]<valve]['Score']

    plt.plot(x, y,label=str(team['TeamName'].iloc[0]))

    



plt.legend() 

plt.show()