import numpy as np, pandas as pd

import os, sys

import matplotlib.pyplot as plt

from pandas.plotting import table

import plotly.express as px

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



INPUT = os.path.join('..', 'input', 'meta-kaggle')

if not os.path.isdir(INPUT):

    INPUT = os.path.join('..', 'input')



# as of early 2019, 211 competitions to show

pd.options.display.max_rows = 250



def do_read_csv(name, index_col=None, dates=None):

    df = pd.read_csv(os.path.join(INPUT, name),

                 index_col=index_col,

                 parse_dates=dates,

                 low_memory=False)

    print (df.shape, name)

    return df
comps = do_read_csv('Competitions.csv', index_col='Id', dates=['DeadlineDate'])

teams = do_read_csv('Teams.csv', index_col='Id')
comps.groupby('HostSegmentTitle').TotalTeams.agg(['count','mean']).sort_values('mean', ascending=False)
include = [ 'Featured', 'Research', 'Recruitment' ]

exclude_slugs = { 'm5-forecasting-accuracy' }



def featured(df):

    idx = df.HostSegmentTitle.isin(include)

    return df.loc[idx]



def comp_filter(df):

    df = featured(df)

    df = df.query('Shakeup>0 and UserRankMultiplier>0 and LeaderboardPercentage>0')

    df = df[~df.Slug.isin(exclude_slugs)]

    return df.copy()



def sort_desc(df, col='Shakeup'):

    return df.sort_values(by=col, ascending=False)
pd.datetime.now().strftime('%c')
sort_desc(featured(comps)[['Title','DeadlineDate']], 'DeadlineDate').head()
comps['Metric'] = comps.EvaluationAlgorithmAbbreviation.str.replace('([a-z])([A-Z])', r'\1 \2')

comps['Year'] = comps.DeadlineDate.dt.year

comps['PublicDataRatio'] = comps.LeaderboardPercentage.astype(str) + '%'
teams['Swing'] = teams.PublicLeaderboardRank - teams.PrivateLeaderboardRank

teams['RankDiff'] = teams['Swing'].abs()
teams.PrivateLeaderboardRank.min()
def ranked(df):

    return df.query('PrivateLeaderboardRank>0')
comp_team_count = ranked(teams).CompetitionId.value_counts()

teams['TeamCount'] = teams.CompetitionId.map(comp_team_count)

teams['RankDiffNorm'] = teams.RankDiff / teams.TeamCount
gb = teams.groupby('CompetitionId')
orig_cols = comps.columns.tolist()
def add_col(name, src, func, int_fill=None):

    comps[name] = gb[src].agg(func)

    if int_fill is not None:

        comps[name] = comps[name].fillna(int_fill).astype(int)
add_col('Entrants', 'RankDiffNorm', 'size')

add_col('TeamCount', 'RankDiffNorm', 'count', int_fill=0)

add_col('Shakeup', 'RankDiffNorm', 'mean')

add_col('WorstDrop', 'Swing', 'min', int_fill=0)

add_col('MedianDelta', 'Swing', 'median', int_fill=0)

add_col('BestRise', 'Swing', 'max', int_fill=0)
shakeup_cols = [c for c in comps.columns if c not in orig_cols]

shakeup_cols
# comps['MeanSubsPerTeam'] = np.round(comps['TotalSubmissions'] / comps['TeamCount'], 1)
comps.to_csv('competitions_with_shakeup.csv')
## by default, show these from original Competitions.csv

base_cols = ['Title']





def link_formatter(r):

    url = f'https://www.kaggle.com/c/{r.Slug}'

    title = ('{Subtitle}'

             '\n'

             '\n'

             'Type: {HostSegmentTitle}'

             '\n'

             'Deadline: {DeadlineDate}'

             '\n'

             'Metric: {Metric}'

             '\n'

             'Public LB size: {PublicDataRatio}').format(**r)

    return f'<a href="{url}" title="{title}">{r.Title}</a>'





def show_comps(cols, sortby='Shakeup'):

    df = comp_filter(comps)

    df['Title'] = df.apply(link_formatter, 1)

    df = df[cols + base_cols]

    df = sort_desc(df, sortby)

    df = df.set_index('Title')

    df.columns = df.columns.str.replace(r'([a-z])([A-Z])', r'\1<br/>\2')

    barcols = [c for c in df.columns if 'shake' in c.lower()]

    return df.style.bar(subset=barcols, color='#20beff')
show_comps(shakeup_cols)
teams['PublicLeaderboardRankRel'] = teams['PublicLeaderboardRank'] / teams['TeamCount']
gb = teams.query('PublicLeaderboardRankRel<0.1').groupby('CompetitionId')
add_col('Shakeup10', 'RankDiffNorm', 'mean')

add_col('WorstDrop10', 'Swing', 'min', int_fill=0)

add_col('MedianDelta10', 'Swing', 'median', int_fill=0)

add_col('BestRise10', 'Swing', 'max', int_fill=0)
shakeup10_cols = [c for c in comps.columns if c not in orig_cols and c not in shakeup_cols]

shakeup10_cols
comps.to_csv('competitions_with_shakeup_and_top10pc_shakeup.csv')
shakeup_cols
show_comps(shakeup_cols[:3] + shakeup10_cols, sortby=['Shakeup10'])
teams.Medal.value_counts()
gb = teams.query('Medal==1').groupby('CompetitionId')   # gold medalists
add_col('ShakeupGold', 'RankDiffNorm', 'mean')

add_col('WorstDropGold', 'Swing', 'min', int_fill=0)

add_col('MedianDeltaGold', 'Swing', 'median', int_fill=0)

add_col('BestRiseGold', 'Swing', 'max', int_fill=0)
comps.Shakeup.count(), comps.ShakeupGold.count()
comps.ShakeupGold.fillna(0, inplace=True)  # some are nan - no gold medals (?)
existing = orig_cols + shakeup_cols + shakeup10_cols

shakeup_gold_cols = [c for c in comps.columns if c not in existing and c != 'Rel10']

shakeup_gold_cols
comps.to_csv('competitions_with_shakeup_x3_final.csv')
# switch order to see more relevant columns

show_comps(shakeup_cols[:3] + shakeup_gold_cols, sortby=['BestRiseGold', 'Shakeup'])
shake_cols = ['Shakeup', 'Shakeup10', 'ShakeupGold']

numeric_columns = comps.select_dtypes(include=['number', bool]).columns

others = [c for c in numeric_columns if c not in shake_cols]

cols = shake_cols + others

comps[cols].rank(pct=True).corr().iloc[:, :len(shake_cols)].style.background_gradient()
plt.rc('figure', figsize=(14, 10))
y = comps.Shakeup

color = np.where(comps.HostSegmentTitle == 'InClass', 'blue', 'red')



s_plots = [

    'Shakeup10',

    'TotalTeams',

    'TotalCompetitors',

    'TotalSubmissions',

    'LeaderboardPercentage',

    'MaxTeamSize',

]



for i, col in enumerate(s_plots, 1):

    plt.subplot(320 + i)

    x = comps[col]

    plt.scatter(x, y, c=color, alpha=0.2)

    plt.ylabel('Shake-up')

    plt.title(col)

plt.tight_layout();
cmap = {

    'Featured': 'blue',

    'Research': 'green',

    'Recruitment': 'red',

    'GE Quests': 'slateblue',

    'Getting Started': 'slateblue',

    'Playground': 'slateblue',

    'Prospect': 'slateblue',

    'InClass': '#9fb',

}



# comps.loc[(comps.HostSegmentTitle != 'InClass')]

fig = px.scatter(comps.dropna(subset=['EvaluationAlgorithmName']),

                 title='Competition Shake-up',

                 x='TotalTeams',

                 y='Shakeup',

                 log_x=True,

                 log_y=True,

                 hover_name='Title',

                 hover_data=[

                     'EvaluationAlgorithmAbbreviation', 'TotalTeams',

                     'TotalSubmissions', 'DeadlineDate'

                 ],

#                  color='EvaluationAlgorithmName'

                 color='HostSegmentTitle',

                 color_discrete_map=cmap

                )

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)
fig = px.scatter(comps.assign(SubsPerTeam=comps.eval('TotalSubmissions / TotalTeams').fillna(0)),

                 title='Competition Shake-up',

                 x='SubsPerTeam',

                 y='Shakeup',

                 log_x=True,

                 log_y=True,

                 hover_name='Title',

                 hover_data=[

                     'EvaluationAlgorithmAbbreviation', 'TotalTeams',

                     'TotalSubmissions', 'DeadlineDate'

                 ],

                 color='HostSegmentTitle',

                 color_discrete_map=cmap)

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)
plt.rc('font', size=12)
def fourDP(s):

    return f'{s:.4f}'



def render(query=None, filename=None, edge_color='#c0c0c0'):

    src = comps if query is None else comps.query(query)

    shakeup = sort_desc(comp_filter(src))

    ss = shake_cols

    shakeup = shakeup[base_cols + ['Metric', 'TeamCount'] + ss].set_index('Title').copy()

    shakeup[ss] = shakeup[ss].applymap(fourDP)

    shakeup['Metric'] = shakeup.Metric.str.replace('[a-z ]', '') # fully abbreviate it

    print(shakeup.shape[0], 'competitions')



    fig, ax = plt.subplots(figsize=(12, 12))

    ax.xaxis.set_visible(False)

    ax.yaxis.set_visible(False)

    ax.set_frame_on(False)

    t = table(ax, shakeup, loc='upper right')

    t.auto_set_font_size(False)

    t.auto_set_column_width(np.arange(shakeup.shape[1]))



    if edge_color is not None:

        for k, cell in t._cells.items():

            cell.set_edgecolor(edge_color)



    if filename is not None:

        plt.savefig(filename, bbox_inches='tight', transparent=True)

    return None
render(query='TeamCount>=100', filename='shakeup_table.png')
render(query='TeamCount>=800', filename='shakeup_table_mini.png')