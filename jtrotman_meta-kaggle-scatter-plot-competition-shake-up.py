import os, sys, subprocess

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core.display import HTML, Image

import plotly.express as px



CSV_DIR = '../input'

OUTPUT_ZIP = 'shakeup_scatter_plots.zip'

X, Y = 'PublicLeaderboardRank', 'PrivateLeaderboardRank'

CODEC = 'png'  # Fork & try 'jpg'!

N_SHOW = 40



def do_read_csv(name):

    df = pd.read_csv(name, low_memory=False)

    print (df.shape, name)

    return df
teams = do_read_csv(f'{CSV_DIR}/Teams.csv')

comps = do_read_csv(f'{CSV_DIR}/Competitions.csv').set_index('Id')

comps['Deadline'] = comps.DeadlineDate.str.split().str[0]

comps['DeadlineDate'] = pd.to_datetime(comps.DeadlineDate)

# InClass are most common, followed by Featured

comps.HostSegmentTitle.value_counts()
exclude_slugs = {'m5-forecasting-accuracy'}

teams = teams.dropna(subset=[X, Y])

n_priv = teams.groupby('CompetitionId').PrivateLeaderboardRank.count()

with_private = n_priv[n_priv > 0].index

comps_idx = (  (comps.LeaderboardPercentage > 0)

             & (comps.index.isin(with_private))

             & (~comps.Slug.isin(exclude_slugs))  )

selected_comps = comps.loc[comps_idx].copy()

selected_comps.shape
teams = teams.loc[teams.CompetitionId.isin(selected_comps.index)]

teams = teams.assign(Medal=teams.Medal.fillna(0).astype(int))

teams.shape
IMAGE_DIR = '/kaggle/plots'

os.makedirs(IMAGE_DIR, exist_ok=True)
shakes = {}

image_files = {}

COLOR_DICT = {0: 'deepskyblue', 1: 'gold', 2: 'silver', 3: 'chocolate'}

savefig_opts = dict(bbox_inches='tight')

plt.rc('font', size=14)

for i, df in teams.groupby('CompetitionId', sort=False):

    fname = comps.Slug[i]

    row = comps.loc[i]

    shakeup = df.eval('abs(PrivateLeaderboardRank-PublicLeaderboardRank)').mean() / df.shape[0]

    title = (f'{row.Title} — {row.TotalTeams} teams — '

             f'{shakeup:.3f} shake-up — {row.Deadline}')

    image_file = f'{IMAGE_DIR}/{fname}.{CODEC}'

    image_files[i] = image_file

    shakes[i] = shakeup

    df = df.sort_values('PrivateLeaderboardRank',

                        ascending=False)  # plot gold last

    ax = df.plot.scatter(X, Y, c=df.Medal.map(COLOR_DICT), figsize=(15, 15))

    plt.title(title, fontsize=16)

    l = np.arange(df.PrivateLeaderboardRank.max())

    ax.plot(l, l, linestyle='--', linewidth=1, color='Black', alpha=0.5)

    ax.set_xlabel(X)

    ax.set_ylabel(Y)

    plt.tight_layout()

    plt.savefig(image_file, **savefig_opts)

    plt.close()
comps['FileSize'] = pd.Series(image_files).apply(os.path.getsize)

comps['Shakeup'] = pd.Series(shakes)

comps.to_csv('CompetitionsWithShakeup.csv')

comps['Image'] = pd.Series(image_files)
def fmt_link(row):

    url = 'https://www.kaggle.com/c/{Slug}'.format(**row)

    txt = '{Deadline} [{EvaluationAlgorithmName}]\n\n{LeaderboardPercentage}% public\n\n{Subtitle}'.format(**row)

    title = row.Title

    return f'<a href="{url}" title="{txt}">{title}</a>'



show = [

    'Title', 'HostSegmentTitle', 'TotalTeams',

    'Deadline', 'Shakeup', 'FileSize'

]

bars = ['TotalTeams', 'Shakeup', 'FileSize']



tmp = comps.assign(Title=comps.apply(fmt_link, 1))

tmp = tmp.sort_values('FileSize', ascending=False)[show]

tmp = tmp.set_index('Title').head(50)

tmp.style.bar(subset=bars)
show = [

    'HasKernels', 'OnlyAllowKernelSubmissions', 'LeaderboardPercentage',

    'MaxDailySubmissions', 'NumScoredSubmissions', 'MaxTeamSize',

    'BanTeamMergers', 'EnableTeamModels', 'EnableSubmissionModelHashes',

    'EnableSubmissionModelAttachments', 'RewardQuantity', 'NumPrizes',

    'UserRankMultiplier', 'CanQualifyTiers', 'TotalTeams', 'TotalCompetitors',

    'TotalSubmissions', 'FileSize', 'Shakeup'

]



plt.rc('font', size=12)

plt.figure(figsize=(14, 12))

sns.heatmap(comps[show].corr(method='spearman'),

            vmin=-1,

            cmap='RdBu',

            annot=True,

            fmt='.1f',

            linewidths=1)

plt.title('Kaggle Competition Attributes - Spearman Correlation');
color_discrete_map = {

     'Featured': 'blue',

     'Research': 'green',

     'Recruitment': 'red',

     'GE Quests': 'slateblue',

     'Getting Started': 'slateblue',

     'Playground': 'slateblue',

     'Prospect': 'slateblue',

}

fig = px.scatter(comps.loc[comps_idx & (comps.HostSegmentTitle != 'InClass')],

                 title='Competition Shake-up',

                 x='Shakeup',

                 y='FileSize',

                 log_y=True,

                 hover_name='Title',

                 hover_data=[

                     'EvaluationAlgorithmAbbreviation', 'TotalTeams',

                     'TotalSubmissions', 'Deadline'

                 ],

                 color='HostSegmentTitle',

                 color_discrete_map=color_discrete_map)

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)
fig = px.scatter(comps.loc[comps_idx & (comps.HostSegmentTitle != 'InClass')],

                 title='Competition Shake-up',

                 x='TotalTeams',

                 y='FileSize',

                 log_x=True,

                 log_y=True,

                 hover_name='Title',

                 hover_data=[

                     'EvaluationAlgorithmAbbreviation', 'Shakeup',

                     'TotalSubmissions', 'Deadline'

                 ],

                 color='HostSegmentTitle',

                 color_discrete_map=color_discrete_map)

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)
# uses a row from Competitions dataframe

def plot(i, row, tag=''):

    pre = 'https://www.kaggle.com'

    html = ('<h1 id="{tag}{Slug}">[#{i}] {Title}</h1>'

            '<h3>{Subtitle}</h3>'

            '<p>[ <a href="{pre}/c/{Slug}">home</a>'

            '   | <a href="{pre}/c/{Slug}/discussion">discussion</a>'

            '   | <a href="{pre}/c/{Slug}/leaderboard">leaderboard</a> ]'

            '<br/>').format(pre=pre, tag=tag, i=i, **row)

    display(HTML(html))

    display(Image(row.Image))



src = comps.sort_values('FileSize', ascending=False)

for i, (row_id, row) in enumerate(src.head(N_SHOW).iterrows(), 1):

    plot(i, row)
by_date = comps.query("HostSegmentTitle!='InClass'")

by_date = by_date.sort_values("DeadlineDate", ascending=False)



for i, (row_id, row) in enumerate(by_date.head(5).iterrows(), 1):

    plot(i, row, 'most-recent-')
Slug = 'mercari-price-suggestion-challenge'
MEDAL_NAMES = np.asarray(["None", "Gold", "Silver", "Bronze"])

MEDAL_COLORS = dict(zip(MEDAL_NAMES,  # depends on Python 3 dict order

                        COLOR_DICT.values()))



selected = comps.query(f"Slug=='{Slug}'")



assert len(selected) == 1, f"Slug {Slug} not found"



chosen = selected.iloc[0]

chosen_teams = teams.query(f"CompetitionId=={chosen.name}").fillna("")

chosen_teams = chosen_teams.assign(Medal=MEDAL_NAMES[chosen_teams.Medal])



# possible improvement: read other fields like # of submissions



fig = px.scatter(chosen_teams,

                 title='Shake-up ' + chosen.Title,

                 x='PublicLeaderboardRank',

                 y='PrivateLeaderboardRank',

                 hover_name='TeamName',

                 hover_data=[

                     'ScoreFirstSubmittedDate',

                     'LastSubmissionDate',

                     'PublicLeaderboardSubmissionId',

                     'PrivateLeaderboardSubmissionId',

                     'Medal',

                     'MedalAwardDate',

                 ],

                 color='Medal',

                 color_discrete_map=MEDAL_COLORS)

fig.update_traces(marker=dict(size=8))

fig.update_layout(height=750, showlegend=False)
!7z a -bd -mmt4 {OUTPUT_ZIP} {IMAGE_DIR}/*.{CODEC}
_ = """

Rerun for recently finished competitions:



    2020-10-04 | Slug:landmark-recognition-2020

    2020-10-09 | Slug:iwildcam-2020-fgvc7

    2020-10-10 | Slug:osic-pulmonary-fibrosis-progression

    2020-10-10 | Slug:stanford-covid-vaccine





"""
_ = """

Rerun for recently finished competitions:



    2020-10-04 | Slug:landmark-recognition-2020

    2020-10-09 | Slug:iwildcam-2020-fgvc7

    2020-10-10 | Slug:osic-pulmonary-fibrosis-progression

    2020-10-10 | Slug:stanford-covid-vaccine





"""