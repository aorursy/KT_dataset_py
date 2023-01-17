import numpy as np, pandas as pd

import os, gc, sys
pd.options.display.max_rows = 300
IN_DIR = '../input'
def read_csv_filtered(csv, col, values):

    dfs = [df.loc[df[col].isin(values)]

           for df in pd.read_csv(csv, chunksize=100000)]

    return pd.concat(dfs, axis=0)
comps = pd.read_csv(f'{IN_DIR}/Competitions.csv', index_col='Id', parse_dates=['DeadlineDate'])

comps.shape
comps['Year'] = comps.DeadlineDate.dt.year
comps.groupby('HostSegmentTitle').TotalTeams.agg(['count','mean']).sort_values('mean', ascending=False)
COMP_TYPES = [ 'Featured', 'Research', 'Recruitment' ]
comps = comps.loc[comps.HostSegmentTitle.isin(COMP_TYPES)].copy()

comps.shape
pd.datetime.now().strftime('%c')
comps[['Title','DeadlineDate']].sort_values('DeadlineDate', ascending=False).head()
teams = read_csv_filtered(f'{IN_DIR}/Teams.csv', 'CompetitionId', comps.index.values)

teams.shape
teams.head()
subs = read_csv_filtered(f'{IN_DIR}/Submissions.csv', 'TeamId', teams.Id.values)

subs.shape
subs.head()
subs = subs.query('not IsAfterDeadline')

subs.shape
used_sub_ids = teams.PrivateLeaderboardSubmissionId.dropna().astype(int).values

used_sub_ids.shape
subs = subs[subs.Id.isin(used_sub_ids)].copy()

subs.shape
subs['CompetitionId'] = subs.TeamId.map(teams.set_index('Id').CompetitionId)
GROUP = [ 'CompetitionId', 'PublicScoreFullPrecision', 'PrivateScoreFullPrecision' ]
subs['Count'] = subs.groupby(GROUP).Id.transform('count')
teams['PrivateEntryCount'] = teams.PrivateLeaderboardSubmissionId.map(subs.set_index('Id').Count)

teams['PrivateEntryCount'].fillna(0, inplace=True)

teams['Shared'] = teams['PrivateEntryCount'] > 1
teams.Medal.value_counts()
MEDALS = [ 'Gold', 'Silver', 'Bronze' ]

FUNCS = [ 'count', 'sum', 'mean' ]
for i, m in enumerate(MEDALS, 1):

    df = teams.query(f'Medal=={i}').groupby('CompetitionId').Shared.agg(FUNCS)

    df.columns = [m, f'{m}_Share',  f'{m}_Frac' ]

    comps = comps.join(df)
comps['Medal_Share'] = comps[['Gold_Share', 'Silver_Share', 'Bronze_Share']].sum(1)
comps.columns
DISPLAY_COLS = ['Year', 'Title', 'TotalTeams', 'Medal_Share',

         'Gold', 'Gold_Share', 'Gold_Frac',

         'Silver', 'Silver_Share', 'Silver_Frac',

         'Bronze', 'Bronze_Share', 'Bronze_Frac'

        ]
def display_comps(querystr=None, sortby='Medal_Share'):

    src = comps[DISPLAY_COLS]

    if querystr is not None:

        src = src.query(querystr)

    if sortby is not None:

        src = src.sort_values(sortby, ascending=False)

    src = src.set_index('Title')

    return src
display_comps('Gold_Share>0')
display_comps('TotalTeams>=500 and Silver_Share>0', sortby='Silver_Frac')
display_comps('Medal_Share>0')
comps.to_csv('competitions_with_shared_medal_stats.csv')
import matplotlib.pyplot as plt

from pandas.plotting import table
def render(query=None):

    src = display_comps(query)

    src = src.dropna().copy()

    roundcols = [ 'Gold_Frac', 'Silver_Frac', 'Bronze_Frac']

    src[roundcols] = src[roundcols].applymap(lambda x: f'{x*100:.0f}%')

    idx = src.dtypes=='float'

    src.loc[:, idx] = src.loc[:, idx].astype(int)

    print(src.shape[0], 'competitions')

    src.columns = src.columns.str.replace('_', ' ')

    plt.clf()

    fig, ax = plt.subplots(figsize=(12, 2))

    ax.xaxis.set_visible(False)

    ax.yaxis.set_visible(False)

    ax.set_frame_on(False)

    t = table(ax, src, loc='upper right')

    t.auto_set_column_width(np.arange(src.shape[1]))

    plt.tight_layout(pad=1)

    return t
render(query='TotalTeams>=2000')
render(query='Year>=2019')
render(query='Year>=2018')
render(query='Year>=2017')
render(query='Medal_Share>0')
comps.groupby('Year').Medal_Share.agg(['count','sum','mean','max'])
comps.groupby('Year').Medal_Share.sum().plot(title='Total Medals Awarded to Teams with Shared Scores')
_ = """

Auto run for competitions:

    [

        'herbarium-2020-fgvc7',

        'plant-pathology-2020-fgvc7',

        'flower-classification-with-tpus',

        'abstraction-and-reasoning-challenge',

        'covid19-global-forecasting-week-5'

    ]



"""