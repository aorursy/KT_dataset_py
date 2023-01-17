from google.cloud import bigquery

import pandas as pd

import bq_helper 

import numpy as np

%matplotlib inline

from wordcloud import WordCloud

import matplotlib.pyplot as plt



import spacy

nlp = spacy.load('en')



# create a helper object for our bigquery dataset

bq_hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")



client = bigquery.Client()

%matplotlib inline
query ="""

SELECT score, author, time

FROM `bigquery-public-data.hacker_news.stories`

WHERE time > 1387536270 AND score >= 0

"""

bq_hacker_news.estimate_query_size(query)
# df_hn = bq_hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# df_hn.to_csv("hn_scores.csv") 
import re

def add_datepart(df, fldname, drop=True, time=False, errors="raise"):	

    """add_datepart converts a column of df from a datetime64 to many columns containing

    the information from the date. This applies changes inplace.

    Parameters:

    -----------

    df: A pandas data frame. df gain several new columns.

    fldname: A string that is the name of the date column you wish to expand.

        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.

    drop: If true then the original date column will be removed.

    time: If true time features: Hour, Minute, Second will be added.

    Examples:

    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })

    >>> df

        A

    0   2000-03-11

    1   2000-03-12

    2   2000-03-13

    >>> add_datepart(df, 'A')

    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed

    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800

    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200

    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600

    """

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Hour']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[n] = getattr(fld.dt, n.lower())

    if drop: df.drop(fldname, axis=1, inplace=True)
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn')

df_hn = pd.read_csv('../input/hacker-news/hn_scores.csv', index_col=0)
df_hn.head()
df_hn = df_hn.sort_values('time').reset_index(drop=True)

df_hn["delta_s"]=df_hn.time.diff()

df_hn = df_hn.dropna()

df_hn.head()
# plt.plot(df_hn.time)
outlier_max=1000

story_rate = 1/(df_hn.delta_s[df_hn.delta_s<outlier_max].mean()) # 1/100 sec

print(f'one story every {1/story_rate:.0f} seconds on average')
plt.hist(df_hn.delta_s[df_hn.delta_s<outlier_max],100, density=True)

t = np.arange(1000)

y = story_rate*np.exp(-story_rate*t)

plt.plot(t,y, label="theoretical from mean rate")

plt.title('Interarrival time for Hacker News')

plt.xlabel("seconds");

plt.ylabel("density");

plt.legend(loc='upper right')

df_hn = pd.read_csv('../input/hacker-news/hn_scores.csv', index_col=0)

df_hn = df_hn.sort_values('time').reset_index()

df_hn["delta_s"]=df_hn.time.diff()

df_hn = df_hn.dropna()

df_hn['datetime'] = pd.to_datetime(df_hn.time, unit='s').dt.tz_localize('GMT').dt.tz_convert('US/Pacific') # in GMT

add_datepart(df_hn,'datetime')
import calendar

dow_dic =dict(enumerate(calendar.day_name))

fig,ax=plt.subplots(figsize=(12,6))

for d in range(7):

    df_hn[df_hn.Dayofweek == d].groupby("Hour").mean().delta_s.plot()

plt.legend(list(dow_dic.values()))

plt.ylabel("stories intervals");

plt.xlabel("Time of day hour (Pacific)")

plt.axvline(x=6, c='0.6', linestyle="dashed")

plt.axvline(x=12,  c='0.6', linestyle="dashed")

index_wk_6_12= (df_hn.Dayofweek<5) & (df_hn.delta_s < 1000) & (df_hn.Hour<12) & (df_hn.Hour>6)

plt.hist(df_hn.delta_s[index_wk_6_12],100, density=True)

story_rate = 1/(df_hn.delta_s[index_wk_6_12].mean()) # 1/100 sec

t = np.arange(1000)

y = story_rate*np.exp(-story_rate*t)

plt.plot(t,y, label="theoretical from mean rate")

plt.xlabel("seconds");

plt.ylabel("density");

plt.title('Interarrival time for Hacker News, constant rate')

plt.legend(loc='upper right')
outlier_max=1000

df_hn_high = df_hn[index_wk_6_12]

story_rate = 1/(df_hn_high.delta_s[df_hn_high.delta_s[index_wk_6_12]<outlier_max].mean()) # 1/100 sec

print(f'one story every {1/story_rate:.0f} seconds on average')
import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from scipy.stats import poisson





flip_events = np.random.binomial(1,story_rate,1000000)



def events_to_intervals(x):

    tmp = np.diff(x.cumsum())

    idx_ev = np.where(tmp == 1)

    return np.diff(idx_ev)[0]



interval_times = events_to_intervals(flip_events)

plt.hist(df_hn.delta_s[index_wk_6_12],100,

         label="HN data", density=True)

plt.hist(interval_times,bins=999,color="r", 

         histtype="step",alpha=0.5,

         range=(1,1000),density=True, label="Bernoulli")

plt.ylabel("probability")

plt.title('Interarrival time for Hacker News, constant rate')

plt.title("distribution of inter-arrival time")

plt.legend(loc='upper right')
flip_events[:10]