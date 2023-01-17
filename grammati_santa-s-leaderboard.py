import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
DATA_PATH = '../input/santaleaderboard/traveling-santa-2018-prime-paths-publicleaderboard.csv'
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime( df['SubmissionDate'])
df.head()
def date_score_scatter(df, cutoff=1e9):
    d = df[df['Score'] <= cutoff]
    plt.figure(figsize=(12,8))
    plt.plot(d['Date'], d['Score'], '.')
date_score_scatter(df)
date_score_scatter(df, 2e6)
date_score_scatter(df, 1.54e6)
date_score_scatter(df, 1.52e6)
def show_teams(df, cutoff):
    d = df[df['Score'] <= cutoff]
    plt.figure(figsize=(20,12))
    best = d[['TeamName','Score']].groupby('TeamName').min().sort_values('Score').index
    args = dict(data=d, x='Date', y='Score', hue='TeamName', hue_order=best, palette='muted')
    sns.lineplot(legend=False, **args)
    sns.scatterplot(legend=('brief' if len(best)<=30 else False), **args)

show_teams(df, 1517500)
show_teams(df, 1516000)
df2 = pd.read_csv('../input/leaderboard20181226/santa-leaderboard-2018-12-26.csv')
df2['Date'] = pd.to_datetime(df2['SubmissionDate'])
show_teams(df2, 1515750)
df3 = pd.read_csv('../input/santaleaderboard20190109/santa-leaderboard-2019-01-09.csv')
df3['Date'] = pd.to_datetime(df3['SubmissionDate'])
show_teams(df3, 1515150)
