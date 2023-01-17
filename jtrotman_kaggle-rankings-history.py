%matplotlib inline

import gc, os, re, sys, time

import calendar

import pandas as pd, numpy as np

from pathlib import Path

from IPython.display import display, HTML



MK = Path(f'../input/meta-kaggle')

DS = Path(f'../input/user-achievements-snapshots')

ID = 'Id'

ATYPES = ['Competitions', 'Scripts', 'Discussion']



pd.options.display.max_rows = 200



TIERNAMES = [

    "novice", "contributor", "expert", "master", "grandmaster", "overlord"

]



DTYPE = {

    'UserId': ('int32'),

    'AchievementType': ('O'),

    'Tier': ('int8'),

    'Points': ('int32'),

    'CurrentRanking': ('float32'),

    'HighestRanking': ('float32'),

    'TotalGold': ('int32'),

    'TotalSilver': ('int32'),

    'TotalBronze': ('int32'),

}
def read_all():

    dfs = []

    for f in sorted(os.listdir(DS)):

        m = re.match(r'UserAchievements_(\d\d)(\d\d)(\d\d).csv', f)

        if m:

            ds = '/'.join(m.groups())

            date = pd.to_datetime(ds, yearfirst=True)

            df = pd.read_csv(DS / f,

                             index_col=0,

                             dtype=DTYPE,

                             parse_dates=['TierAchievementDate'])

            dfs.append(df.assign(date=date))

    return pd.concat(dfs)
df = read_all()

df.shape
users = pd.read_csv(MK / 'Users.csv', index_col=0)

users.shape
# Add 1 to the codes - the number now refers to the number of dots in the UI

df['TierName'] = df.Tier.apply(lambda x: f't{x+1}_{TIERNAMES[x]}')
df.head()
df.loc[df.AchievementType=="Competitions"].groupby(['date', 'TierName']).size().unstack().style.background_gradient(axis=1)
df.loc[df.AchievementType=="Scripts"].groupby(['date', 'TierName']).size().unstack().style.background_gradient(axis=1)
df.loc[df.AchievementType=="Discussion"].groupby(['date', 'TierName']).size().unstack().style.background_gradient(axis=1)
TCOLS = [

    't1_novice', 't2_contributor', 't3_expert', 't4_master', 't5_grandmaster'

]
import matplotlib as mpl

import matplotlib.pyplot as plt



FIGSIZE = (15, 9)

plt.style.use('ggplot')

plt.rc('figure', figsize=FIGSIZE)  # works locally, not on Kaggle

plt.rc('font', size=14)
for t in ATYPES:

    display(HTML(f"<h2>{t} Tier Counts Over Time</h2>"))

    tmp = pd.get_dummies(df.loc[df.AchievementType==t], columns=['TierName'], prefix='', prefix_sep='')

    tmp.groupby('date')[TCOLS].sum().plot(figsize=FIGSIZE)

    plt.show()
MIN_TIER = 2

for t in ATYPES:

    display(HTML(f"<h2>{t} Tier Counts Over Time</h2>"))

    tmp = df.loc[(df.AchievementType==t) & (df.Tier>=MIN_TIER)]

    tmp = pd.get_dummies(tmp, columns=['TierName'], prefix='', prefix_sep='')

    tmp.groupby('date')[TCOLS[MIN_TIER:]].sum().plot(figsize=FIGSIZE)

    plt.show()
MIN_TIER = 3

for t in ATYPES:

    display(HTML(f"<h2>{t} Tier Counts Over Time</h2>"))

    tmp = df.loc[(df.AchievementType==t) & (df.Tier>=MIN_TIER)]

    tmp = pd.get_dummies(tmp, columns=['TierName'], prefix='', prefix_sep='')

    tmp.groupby('date')[TCOLS[MIN_TIER:]].sum().plot(figsize=FIGSIZE)

    plt.show()
MIN_TIER = 4

for t in ATYPES:

    display(HTML(f"<h2>{t} Tier Counts Over Time</h2>"))

    tmp = df.loc[(df.AchievementType==t) & (df.Tier>=MIN_TIER)]

    tmp = pd.get_dummies(tmp, columns=['TierName'], prefix='', prefix_sep='')

    tmp.groupby('date')[TCOLS[MIN_TIER:]].sum().plot(figsize=FIGSIZE)

    plt.show()
def show_top(ranks):

    for t in ATYPES:

        title = f"{t} Points at Top {ranks} Ranks"

        display(HTML(f"<h2>{t}</h2>"))

        for r in ranks:

            tmp = df.loc[(df.AchievementType == t) & (df.CurrentRanking == r)].set_index('date')

            tmp.Points.plot(figsize=FIGSIZE, label=f'Top {r}', legend=True)

        plt.ylim(0)

        plt.title(title)

        plt.show()



def show_top_sum(fields, func):

    for t in ATYPES:

        title = f"{t} {func} {fields}"

        display(HTML(f"<h2>{t}</h2>"))

        tmp = df.loc[(df.AchievementType == t)]

        res = tmp.groupby('date')[fields].agg(func)

        res.plot(figsize=FIGSIZE, label=f'{func} {fields}', legend=True)

        plt.ylim(0)

        plt.title(title)

        plt.show()
show_top([100])
show_top([1, 5, 10])
show_top([200, 400])
show_top_sum(['TotalGold', 'TotalSilver', 'TotalBronze'], 'sum')
show_top_sum(['TotalGold', 'TotalSilver'], 'sum')
show_top_sum('TotalGold', 'sum')