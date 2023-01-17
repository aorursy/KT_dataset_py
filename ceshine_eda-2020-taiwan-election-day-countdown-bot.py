!pip install -U altair vega_datasets
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import glob

from pathlib import Path



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt



for dirname, _, filenames in os.walk('/kaggle/input/statistics-2020-taiwan-election-day-countdown-bot/election-2020-01-11/'):

    for filename in filenames:

        print((Path(dirname) / filename).stem)
pattern = "/kaggle/input/statistics-2020-taiwan-election-day-countdown-bot/election-2020-01-11/*.csv"

df_all = pd.concat((pd.read_csv(x) for x in glob.glob(pattern)), axis=0).reset_index(drop=True)

df_all.columns
df_all[['Tweet text', 'time', 'impressions', 'engagements', 'engagement rate', 

        'retweets', 'replies', 'likes', 'user profile clicks', 'url clicks', 

        'hashtag clicks', 'follows']].sample(2)
df_all["time"].sort_values().tail()
df_all["time"] = pd.to_datetime(pd.to_datetime(df_all["time"]).dt.tz_convert('Asia/Taipei').dt.date)
source = df_all[["time", "impressions"]]

bar = alt.Chart(source).mark_bar().encode(

    x='time:T',

    y='impressions:Q'

).properties(width=600)

bar
from datetime import datetime

bar = alt.Chart(source[source["time"] >= "2019-11-01"]).transform_calculate(

    electionDate="date(datum.time) == 11 & month(datum.time) == 0",

).mark_bar().encode(

    x='time:T',

    y='impressions:Q',

    color=alt.condition(

        "datum.electionDate",  

        alt.value('orange'),     # which sets the bar orange.

        alt.value('steelblue')   # And if it's not true it sets the bar steelblue.

    )

).properties(width=600)

bar
df_all[['Tweet text', 'time', 'impressions', 'engagements', 'engagement rate', 

        'retweets', 'replies', 'likes', 'user profile clicks', 'url clicks', 

        'hashtag clicks', 'follows']].sort_values("time").tail(2)