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
users = pd.read_csv(MK / 'Users.csv', index_col=0)

dsv = pd.read_csv(MK / 'DatasetVersions.csv', index_col=0)

datasets = pd.read_csv(MK / 'Datasets.csv', index_col=0)
idx = datasets.OwnerUserId.isnull()

datasets.loc[idx, 'OwnerUserId'] = datasets.loc[idx, 'CreatorUserId']
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
srcs = pd.read_csv(MK / 'KernelVersionDatasetSources.csv', index_col=ID)

srcs['ScriptId'] = srcs.KernelVersionId.map(versions.ScriptId)

scripts = srcs.groupby('ScriptId').SourceDatasetVersionId.nunique().to_frame('UniqueDatasets')

# discard those using more than one dataset source

scripts = scripts.query('UniqueDatasets==1').copy()

scripts['SourceDatasetVersionId'] = srcs.groupby('ScriptId').SourceDatasetVersionId.min()

scripts['CurrentUrlSlug'] = scripts.index.map(kern.CurrentUrlSlug)

scripts['TotalVotes'] = scripts.index.map(versions.groupby('ScriptId').Votes.sum()).astype(int)

scripts['TotalViews'] = scripts.index.map(kern.TotalViews)

scripts['Date'] = scripts.index.map(kern.Date)

scripts.shape
count = 0

for cid, df in scripts.groupby('SourceDatasetVersionId'):

    nb = df.shape[0]

    if nb < 35: continue



    df = df.sort_values('TotalVotes', ascending=False).head(SHOW_TOP)

    df = df.assign(KernelName=df.CurrentUrlSlug + " [" + df.TotalVotes.map(str) + "]")

    ds = dsv.loc[cid]

    d = (DATE_THRES - pd.to_datetime(ds.CreationDate)).days

    t = 'past' if d > 0 else 'to go'

    d = abs(d)

    uid = datasets.loc[ds.DatasetId].OwnerUserId

    if uid not in users.index:

        continue

    u = users.loc[uid].UserName

    

    display(

        HTML(

            f"<h1 id={ds.Slug}>{ds.Title}</h1> "

            f"<p>Creation Date: {ds.CreationDate} ({d} days {t})"

            f"<p><a href='{HOST}/{u}/{ds.Slug}/kernels'>Kernels Listing</a> ({nb} kernels)"

        ))



    sns.set(rc={'figure.figsize': FIGSIZE})

    sns.set(font_scale=1.1)

    sns.boxplot(data=df.join(daily).set_index('KernelName')[VCOLS].T, orient='h')

    plt.title(ds.Title, loc='left')

    plt.xlabel("Daily Notebook Views")

    plt.tight_layout()

    plt.show()

    count += 1
print(f'{count} datasets shown')