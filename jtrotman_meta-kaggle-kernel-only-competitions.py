import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import HTML, Image, display



def read_csv_filtered(csv, col, values, **kwargs):

    dfs = [df.loc[df[col].isin(values)]

           for df in pd.read_csv(csv, chunksize=100000, low_memory=False, **kwargs)]

    return pd.concat(dfs, axis=0)



IN_DIR = '../input'



DATE_COLS = [

    'EnabledDate', 'DeadlineDate', 'ProhibitNewEntrantsDeadlineDate',

    'TeamMergerDeadlineDate', 'TeamModelDeadlineDate',

    'ModelSubmissionDeadlineDate'

]

comps = pd.read_csv(f'{IN_DIR}/Competitions.csv',

                    parse_dates=DATE_COLS)

comps['Days'] = (comps.DeadlineDate - comps.EnabledDate).dt.days

comps.shape
comps.columns
ko = comps.query('OnlyAllowKernelSubmissions').sort_values('DeadlineDate')

ko.shape
show = ['Slug', 'DeadlineDate', 'Days', 'TotalTeams', 'RewardType', 'RewardQuantity']



def make_clickable(val):

    return f'<a href="https://www.kaggle.com/c/{val}">{val}</a>'



ko.set_index('Title')[show].fillna('').style.format(make_clickable, subset='Slug')
teams = read_csv_filtered(f'{IN_DIR}/Teams.csv', 'CompetitionId', ko.Id).set_index('Id')

teams.shape
subs = read_csv_filtered(f'{IN_DIR}/Submissions.csv',

                         'TeamId',

                         teams.index,

                         parse_dates=['SubmissionDate', 'ScoreDate']).set_index('Id')

subs.shape
plt.rc('font', size=14)   



for cid, df in teams.groupby('CompetitionId'):

    comp = comps.query('Id==@cid').iloc[0]

    sdf = subs[subs.TeamId.isin(df.index) & (subs.SubmissionDate <= comp.DeadlineDate)]

    cdf = sdf.groupby('SubmissionDate').agg({'TeamId':['size', 'nunique']})

    cdf.columns = ['Submissions', 'Unique Teams']

    display(HTML(

        f'<h1 id="{comp.Slug}">{comp.Title}</h1>'

        f'<ul>'

        f'<li>Deadline Date: {comp.DeadlineDate}'

        f'<li>Teams Ranked: {df.PublicLeaderboardRank.count()}'

        f'<li>Team Count: {df.shape[0]}'

        f'<li>Submission Count: {sdf.shape[0]}'

        f'</ul>'

        )

    )

    cdf.plot(figsize=(15, 9), title=f'{comp.Title} — Daily Submissions')

    plt.ylabel('Count')

    plt.grid(True, axis='both')

    plt.show()