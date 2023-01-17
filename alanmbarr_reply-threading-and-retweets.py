import numpy as np

import pandas as pd

import seaborn as sns

from subprocess import check_output

df = pd.read_csv("../input/tweets.csv")

f_hc = df.loc[(df['handle']=='HillaryClinton'),['in_reply_to_screen_name']]

f_dt = df.loc[(df['handle']=='realDonaldTrump'),['in_reply_to_screen_name']]
f_dt.apply(pd.value_counts)
f_hc.apply(pd.value_counts)
df = pd.read_csv("../input/tweets.csv")

rt_hc = df.loc[(df['handle']=='HillaryClinton'), ['is_retweet']]

rt_dt = df.loc[(df['handle']=='realDonaldTrump'), ['is_retweet']]
ax = sns.countplot(rt_hc['is_retweet'])

ax.set(xticklabels=["Tweets","Retweets"])

#float_formatter = lambda x: "%.0f" x

hc_rt_count = np.bincount(rt_hc['is_retweet'])[1]

hc_t_count = np.bincount(rt_hc['is_retweet'])[0]

hc_t_total = hc_t_count + hc_rt_count

hc_rt_pct = hc_rt_count / hc_t_total

hc_pt = np.around(hc_rt_pct, decimals=2)

print("Hillary Clinton retweeted {} times, {}% of all tweets".format(hc_rt_count,str(hc_pt).lstrip("0.")))
ax = sns.countplot(rt_dt['is_retweet'])

ax.set(xticklabels=["Tweets","Retweets"])

dt_rt_count = np.bincount(rt_dt['is_retweet'])[1]

dt_t_count = np.bincount(rt_dt['is_retweet'])[0]

dt_t_total = dt_t_count + dt_rt_count

dt_rt_pct = dt_rt_count / dt_t_total

dt_pt = np.around(dt_rt_pct, decimals=2)

print("Donald Trump's retweeted {} times, {}% of all tweets".format(dt_rt_count,str(dt_pt).lstrip("0.")))