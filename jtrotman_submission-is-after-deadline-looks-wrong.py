import gc, os, sys, time

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

from IPython.display import HTML, display



IN_DIR = os.path.join('..', 'input', 'meta-kaggle')



comps = pd.read_csv(os.path.join(IN_DIR, 'Competitions.csv'))

comps['DeadlineDate'] = pd.to_datetime(comps.DeadlineDate)

comps['EnabledDate'] = pd.to_datetime(comps.EnabledDate)

comps['Days'] = (comps.DeadlineDate - comps.EnabledDate) / pd.Timedelta(1, 'd')

comps['FinalWeek'] = (comps.DeadlineDate - pd.Timedelta(1, 'w'))

comps.shape
pd.read_csv(os.path.join(IN_DIR, 'Teams.csv'), nrows=4).columns
teams = pd.read_csv(os.path.join(IN_DIR, 'Teams.csv'), usecols=['Id', 'CompetitionId'])

teams.shape
pd.read_csv(os.path.join(IN_DIR, 'Submissions.csv'), nrows=4).columns
subs = pd.read_csv(os.path.join(IN_DIR, 'Submissions.csv'), usecols=['Id', 'TeamId', 'SubmissionDate', 'ScoreDate', 'IsAfterDeadline'])

subs.shape
subs['SubmissionDate'] = pd.to_datetime(subs.SubmissionDate)
subs['CompetitionId'] = subs.TeamId.map(teams.set_index('Id').CompetitionId)
subs['DeadlineDate'] = subs.CompetitionId.map(comps.set_index('Id').DeadlineDate)
subs['DeadlineDiff'] = (subs['SubmissionDate'] - subs['DeadlineDate']) / pd.Timedelta(1, 'd')
subs.count()
comp = comps[comps.Slug.str.startswith('mercedes')].squeeze()

comp_id = comp.Id
merc = subs.query(f'CompetitionId=={comp_id}')

merc.shape
merc.groupby('IsAfterDeadline').size()
merc.groupby('IsAfterDeadline').SubmissionDate.agg(['min','max'])
merc.groupby(['SubmissionDate', 'IsAfterDeadline']).size().unstack().head(55)
plt.rc("figure", figsize=(12, 10))

plt.rc("font", size=12)
def colors(df):

    return np.where(df['IsAfterDeadline'], 'red', 'blue')
merc.plot.scatter('SubmissionDate', 'TeamId', c=colors(merc), alpha=.1, title=comp.Title);
m1 = merc.query('DeadlineDiff<=21')

m1.plot.scatter('SubmissionDate', 'TeamId', c=colors(m1), alpha=.1, title=comp.Title);
m1 = merc.query('DeadlineDiff<=21')

m1.plot.scatter('Id', 'TeamId', c=colors(m1), alpha=.1, title=comp.Title);
comps = comps.set_index('Id')
comps.nlargest(20, 'TotalTeams')[['Slug', 'Title', 'DeadlineDate', 'TotalTeams']]
THRES = 3000
for comp_id, subset in subs.groupby('CompetitionId'):

    if comp_id not in comps.index: # KeyError: 23099 for "SIIM-ISIC Melanoma Classification" ?

        continue

    comp = comps.loc[comp_id]

    if comp.TotalTeams < THRES:

        continue

    window = subset.query('DeadlineDiff<=21')

    

    markup = (

        '<h1 id="{Slug}">{Title}</h1>'

        '<p>'

        'Type: {HostSegmentTitle} &mdash; <i>{Subtitle}</i>'

        '<br/>'

        '<a href="https://www.kaggle.com/c/{Slug}/leaderboard">Leaderboard</a>'

        '<br/>'

        'Dates: <b>{EnabledDate}</b> &mdash; <b>{DeadlineDate}</b>'

        '<br/>'

        '<b>{TotalTeams}</b> teams; <b>{TotalCompetitors}</b> competitors; '

        '<b>{TotalSubmissions}</b> submissions'

        '<br/>').format(**comp)



    display(HTML(markup))

    window.plot.scatter('SubmissionDate', 'TeamId', c=colors(window), alpha=.1, title=comp.Title)

    plt.show()