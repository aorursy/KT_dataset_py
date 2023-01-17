%matplotlib inline

import gc, os, sys, time

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from IPython.display import HTML, display



IN_DIR = '../input'

AUC = 'Area Under Receiver Operating Characteristic Curve'



def read_csv_filtered(csv, col, values, **kwargs):

    dfs = [df.loc[df[col].isin(values)]

           for df in pd.read_csv(f'{IN_DIR}/{csv}',

                                 chunksize=100000, **kwargs)]

    return pd.concat(dfs, axis=0)



TYPES = { 'Featured', 'Research', 'Recruitment' }



# Read Competitions

comps = read_csv_filtered('Competitions.csv', 'HostSegmentTitle', TYPES).set_index('Id')

idx = comps.EvaluationAlgorithmName.isnull()

comps.loc[idx, 'EvaluationAlgorithmName'] = comps.loc[idx, 'EvaluationAlgorithmAbbreviation']

comps['Year'] = pd.to_datetime(comps.DeadlineDate).dt.year

comps['RewardQuantity'].fillna('', inplace=True)



comps = comps.query('EvaluationAlgorithmName==@AUC', engine='python')

print("Competitions", comps.shape)



# Read Teams for those Competitions

asints = ['PublicLeaderboardSubmissionId',

           'PrivateLeaderboardSubmissionId']

DT = {c:'object' for c in asints}

teams = read_csv_filtered('Teams.csv', 'CompetitionId', comps.index, dtype=DT).set_index('Id')

teams[asints] = teams[asints].fillna(-1).astype(int)

print("Teams", teams.shape)



asfloats = ['PublicScoreLeaderboardDisplay',

            'PublicScoreFullPrecision',

            'PrivateScoreLeaderboardDisplay',

            'PrivateScoreFullPrecision',]



# Read Submissions for those Teams

subs = read_csv_filtered('Submissions.csv', 'TeamId', teams.index)

subs = subs.query('not IsAfterDeadline', engine='python')

subs[asfloats] = subs[asfloats].astype(float)

print("Submissions", subs.shape)



subs['CompetitionId'] = subs.TeamId.map(teams.CompetitionId)

subs['UsedPublic'] = subs.Id.isin(teams.PublicLeaderboardSubmissionId)

subs['UsedPrivate'] = subs.Id.isin(teams.PrivateLeaderboardSubmissionId)

subs['Used'] = subs.eval('UsedPublic or UsedPrivate')



subs['dist1'] = ((subs.PublicScoreFullPrecision-0.25)**2 + (subs.PrivateScoreFullPrecision-0.75)**2) ** 0.5

subs['dist2'] = ((subs.PublicScoreFullPrecision-0.75)**2 + (subs.PrivateScoreFullPrecision-0.25)**2) ** 0.5



COLORS = np.asarray([

    '#c0c0c080', # gray: not used for either leaderboard

    '#0000ffff', # blue: used as public lb entry for team

    '#ff0000ff', # red: used as private lb entry for team

    '#ff00ffff'  # purple: used as public and private lb entry for team

])

RADIUS = 0.12



subs['Color'] = (subs.UsedPublic*1) + (subs.UsedPrivate*2)

# subs['Color'].value_counts()



# Plot each Competition

for comp, df in subs.groupby('CompetitionId'):

    if df.PublicScoreFullPrecision.var() == 0 or df.PrivateScoreFullPrecision.var() == 0:

        continue

    c = comps.loc[comp]

    if c.Year < 2013:  # bug not there 2010..2012

        continue

    a = (df.dist1 <= RADIUS).sum()

    b = (df.dist2 <= RADIUS).sum()

    used_flippable = df.eval('UsedPrivate and PrivateScoreFullPrecision<0.5').sum()

    top_flipzone = df.eval('UsedPrivate and PrivateScoreFullPrecision<0.75').sum()

    teams = c.TotalTeams # should == df.UsedPrivate.sum()

    display(HTML(

        f'<h1 id="{c.Slug}">{c.Title}</h1><h3>{c.Subtitle}</h3>'

        f'<p>{teams} teams &mdash; {c.TotalSubmissions} submissions.<br/>'

        f'public low ~0.25 &rarr; private high ~0.75 = {a}<br/>'

        f'public high ~0.75 &rarr; private low ~0.25 = {b}<br/>'

        f'private LB teams under 0.5 = {used_flippable}<br/>'

        f'private LB teams under 0.75 = {top_flipzone}<br/>'

    ))

    

    title = f'{c.Title} — {teams} teams — {c.TotalSubmissions} submissions — {c.Year}'

    df1 = df.sort_values('Used')

    ax = df1.plot.scatter('PublicScoreFullPrecision', 'PrivateScoreFullPrecision', c=COLORS[df1.Color], title=title, figsize=(14, 14))

    

    # https://stackoverflow.com/questions/4143502/how-to-do-a-scatter-plot-with-empty-circles-in-python

    for centre in [ (0.25, 0.75), (0.75, 0.25) ]:

        e = Circle(xy=centre, radius=RADIUS)

        ax.add_artist(e)

        e.set_clip_box(ax.bbox)

        e.set_edgecolor('black')

        e.set_facecolor('none')  # "none" not None

        e.set_alpha(0.5)



    plt.axis('equal')

    plt.show()
subs.dist1.plot.hist(bins=100)
subs.dist2.plot.hist(bins=100)
subs.dist1[subs.dist1<0.3].plot.hist(bins=100)
subs.dist2[subs.dist2<0.3].plot.hist(bins=100)