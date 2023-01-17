%matplotlib inline

import gc, os, sys, time

import calendar

import pandas as pd, numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

from IPython.display import HTML, Image, display



YEAR = 2019

MONTH = 11

TAG = f'{YEAR:04d}-{MONTH:02d}'

MK = Path(f'../input/meta-kaggle')

DS = Path(f'../input/kaggle-view-counts-{TAG}')

ID = 'Id'

SHOW_TOP = 20

FIGSIZE = (9, 9)

HOST = 'https://www.kaggle.com'

DAYNAMES = np.asarray(calendar.day_name)

DATE_THRES = pd.to_datetime(TAG)   # count votes before this date
comps = pd.read_csv(MK / 'Competitions.csv', index_col=0)
kernelsDailyTotals = pd.read_csv(DS / 'KernelsDailyTotals.csv', index_col=ID)

kernelsDailyTotals.shape
np.where(kernelsDailyTotals.count()==0)
kernelsDailyTotals['TotalViews7'] = ((kernelsDailyTotals.TotalViews6 + kernelsDailyTotals.TotalViews8) / 2).round(0)
daily = kernelsDailyTotals.diff(axis=1).iloc[:, 1:].dropna(how='all')

VCOLS = daily.columns.str.replace('Total', '')

daily.columns = VCOLS

VCOLS
# daily view counts for 5 kernels

daily.sample(n=5, random_state=2022).T.style.background_gradient(axis=None)
votes = pd.read_csv(MK / 'KernelVotes.csv', parse_dates=['VoteDate'], index_col=ID)

votes.shape
# use only votes from before month started

versions = pd.read_csv(MK / 'KernelVersions.csv', usecols=range(9), index_col=ID)

versions['Votes'] = votes[(votes.VoteDate < DATE_THRES)].KernelVersionId.value_counts()

versions.shape
kern = pd.read_csv(MK / 'Kernels.csv', index_col=ID)

kern['Date'] = pd.to_datetime(kern.CreationDate.str[:10])

kern.shape
srcs = pd.read_csv(MK / 'KernelVersionCompetitionSources.csv', index_col=ID)

srcs['ScriptId'] = srcs.KernelVersionId.map(versions.ScriptId)

scripts = srcs.groupby('ScriptId').SourceCompetitionId.nunique().to_frame('UniqueComps')

# discard those using more than one competition as data source

scripts = scripts.query('UniqueComps==1').copy()

scripts['CompetitionId'] = srcs.groupby('ScriptId').SourceCompetitionId.min()

scripts['CurrentUrlSlug'] = scripts.index.map(kern.CurrentUrlSlug)

scripts['TotalVotes'] = scripts.index.map(versions.groupby('ScriptId').Votes.sum()).astype(int)

scripts['TotalViews'] = scripts.index.map(kern.TotalViews)

scripts['Date'] = scripts.index.map(kern.Date)

scripts.shape
count = 0

for cid, df in scripts.groupby('CompetitionId'):

    nb = df.shape[0]

    if nb < SHOW_TOP: continue



    df = df.sort_values('TotalVotes', ascending=False).head(SHOW_TOP)

    df = df.assign(NotebookName=df.CurrentUrlSlug + " [" + df.TotalVotes.map(str) + "]")

    c = comps.loc[cid]

    d = (DATE_THRES - pd.to_datetime(c.DeadlineDate)).days

    t = 'past' if d > 0 else 'to go'

    d = abs(d)



    display(

        HTML(

            f"<h1 id={c.Slug}>{c.Title}</h1> "

            f"<p>Deadline {c.DeadlineDate} ({d} days {t})"

            f"<p><a href='{HOST}/c/{c.Slug}/notebooks'>Notebook Listing</a> ({nb} notebooks)"

        ))



    sns.set(rc={'figure.figsize': FIGSIZE})

    sns.set(font_scale=1.1)

    sns.boxplot(data=df.join(daily).set_index('NotebookName')[VCOLS].T, orient='h')

    plt.title(c.Title, loc='left')

    plt.xlabel("Daily Notebook Views")

    plt.tight_layout()

    plt.show()

    count += 1
print(f'{count} competitions shown')