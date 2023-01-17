%matplotlib inline

import gc, os, sys, time

import calendar

import pandas as pd, numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from pathlib import Path



YEAR = 2019

MONTH = 11

TAG = f'{YEAR:04d}-{MONTH:02d}'

MK = Path(f'../input/meta-kaggle')

DS = Path(f'../input/kaggle-view-counts-{TAG}')

ID = 'Id'

FIGSIZE = (15,9)

DAYNAMES = np.asarray(calendar.day_name)



plt.style.use('ggplot')

plt.rc('figure', figsize=FIGSIZE) # works locally, not on Kaggle

plt.rc('font', size=14)



pd.options.display.max_rows = 200
K = pd.read_csv(MK / 'Kernels.csv', index_col=ID)

K.shape
KV = pd.read_csv(DS / 'KernelsDailyTotals.csv', index_col=ID)

KV.shape
DV = pd.read_csv(DS / 'DatasetsDailyTotals.csv', index_col=ID)

DV.shape
def make_month_index(year, month):

    d1 = pd.to_datetime(f'{year:04d}-{month:02d}-01 12:00',

                        dayfirst=False,

                        yearfirst=True)

    day = pd.Timedelta(1, 'd')

    days = {0: d1 - day}

    dx = d1

    while dx.month == d1.month:

        days[dx.day] = dx

        dx += day

    return days



DATES_DICT = make_month_index(YEAR, MONTH)
np.where(KV.count()==0)
KV['TotalViews7'] = ((KV.TotalViews6 + KV.TotalViews8) / 2).round(0)
np.where(DV.count()==0)
DV['TotalViews7'] = ((DV.TotalViews6 + DV.TotalViews8) / 2).round(0)

DV['TotalDownloads7'] = ((DV.TotalDownloads6 + DV.TotalDownloads8) / 2).round(0)
daily = KV.diff(axis=1).iloc[:, 1:].dropna(how='all')

cols = daily.columns.str.replace('Total', '')

cols = cols.str.extract('(\d+)')[0].rename('Date').astype(int).map(DATES_DICT)

daily.columns = cols
daily.sum(0).plot(title=f'Daily Kernel Views {TAG}', figsize=FIGSIZE)
daily.sum(0).to_frame('Total Kernel Views').style.background_gradient()
tmp = daily.sum(0).to_frame('Kernel Views').reset_index()

daystats = tmp.groupby(tmp.Date.dt.dayofweek).mean()

daystats.index = DAYNAMES[daystats.index]

daystats.style.background_gradient()
daily.min(1).sort_values(ascending=False).head(20).to_frame('Minimum Daily Views').style.background_gradient()
daily.mean(0).plot(title='Mean Kernel views per day', figsize=FIGSIZE)
daily.shape
daily.max(1).clip(upper=10).value_counts()
daily.max(1).clip(upper=5).value_counts(True) * 100
daily.max(1).clip(upper=10).plot.hist(title='Max of daily view counts for Kernels', figsize=FIGSIZE)