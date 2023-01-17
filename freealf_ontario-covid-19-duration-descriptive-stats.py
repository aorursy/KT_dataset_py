import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
DATE_FIELDS = ['Accurate_Episode_Date','Case_Reported_Date','Test_Reported_Date','Specimen_Date']
latest = pd.read_csv('https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv',

                    parse_dates=DATE_FIELDS,

                    )

latest.head()
percentiles = [50,80,90,95,100]

metrics = ['Episode_to_Report', 'Episode_to_Specimen', 'Specimen_to_Result', 'Result_to_Report']

combo_metrics = ['%s_%d' % (m, p) for m in metrics for p in percentiles]
latest['Episode_to_Report'] = (latest['Case_Reported_Date'] - latest['Accurate_Episode_Date']).dt.days

latest['Episode_to_Specimen'] = (latest['Specimen_Date'] - latest['Accurate_Episode_Date']).dt.days

latest['Specimen_to_Result'] = (latest['Test_Reported_Date'] - latest['Specimen_Date']).dt.days

latest['Result_to_Report'] = (latest['Case_Reported_Date'] - latest['Test_Reported_Date']).dt.days
latest[metrics].describe()
latest[latest['Result_to_Report']==-83]
latest[latest['Episode_to_Specimen']==-89]
latest[latest['Specimen_to_Result']==-30]
for m in metrics:

    print('There are %d cases of missing %s' % (len(latest[np.isnan(latest[m])]), m))

print('... out of %d total cases' % len(latest))
latest_date = latest[DATE_FIELDS].max().max()

latest_date
delay_df = pd.DataFrame(index=pd.date_range('2020-03-01', latest_date), columns=combo_metrics)



for crd, grp in latest[latest['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Case_Reported_Date'):

    for m in metrics:

        for p in percentiles:

            delay_df.loc[crd, '%s_%d' % (m, p)] = grp[m].quantile(p/100)

delay_df.tail()
fig, axarr = plt.subplots(4, figsize=(6, 12))

for i, m in enumerate(metrics):

    ax = axarr[i]

    for p in percentiles:

        delay_df.plot(y='%s_%d' % (m, p), label='%dth percentile' % p, ax=ax)

        ax.set_ylabel(m)

    ax.set_xlabel('Case_Reported_Date')
latest[latest['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Reporting_PHU')[metrics].quantile([0.5, 0.9, 0.95, 1.0]).unstack()
sentinel_date = pd.Timestamp.max

sentinel_date
latest_sentinel = latest.copy()

for d_f in DATE_FIELDS:

    latest_sentinel[d_f] = latest_sentinel[d_f].fillna(sentinel_date)
latest_sentinel['Episode_to_Report'] = (latest_sentinel['Case_Reported_Date'] - latest_sentinel['Accurate_Episode_Date']).dt.days

latest_sentinel['Episode_to_Specimen'] = (latest_sentinel['Specimen_Date'] - latest_sentinel['Accurate_Episode_Date']).dt.days

latest_sentinel['Specimen_to_Result'] = (latest_sentinel['Test_Reported_Date'] - latest_sentinel['Specimen_Date']).dt.days

latest_sentinel['Result_to_Report'] = (latest_sentinel['Case_Reported_Date'] - latest_sentinel['Test_Reported_Date']).dt.days
sentinel_delay_df = pd.DataFrame(index=pd.date_range('2020-03-01', latest_date), columns=combo_metrics)



for crd, grp in latest_sentinel[latest_sentinel['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Case_Reported_Date'):

    for m in metrics:

        for p in percentiles:

            sentinel_delay_df.loc[crd, '%s_%d' % (m, p)] = grp[m].quantile(p/100)

sentinel_delay_df.tail()
fig, axarr = plt.subplots(4, figsize=(6, 12))

for i, m in enumerate(metrics):

    ax = axarr[i]

    for p in percentiles:

        sentinel_delay_df.plot(y='%s_%d' % (m, p), label='%dth percentile' % p, ax=ax)

        ax.set_ylabel(m)

    ax.set_xlabel('Case_Reported_Date')
def correct_sentinel(val_in_days):

    # if it's off by more than 2 years, it's due to missing data

    if (val_in_days > 730) or (val_in_days < -730):

        return np.nan

    else:

        return val_in_days
for cm in combo_metrics:

    sentinel_delay_df[cm] = sentinel_delay_df[cm].apply(correct_sentinel)
fig, axarr = plt.subplots(4, figsize=(6, 12))

for i, m in enumerate(metrics):

    ax = axarr[i]

    for p in percentiles:

        sentinel_delay_df.plot(y='%s_%d' % (m, p), label='%dth percentile' % p, ax=ax)

        ax.set_ylabel(m)

    ax.set_xlabel('Case_Reported_Date')
latest_sentinel[latest_sentinel['Accurate_Episode_Date']>=pd.to_datetime('2020-03-01')].groupby('Reporting_PHU')[metrics].quantile([0.5, 0.9, 0.95, 1.0]).unstack().applymap(correct_sentinel)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session