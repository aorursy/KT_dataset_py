# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/strava-jeddah-segments-leaderboard/jeddah_strava_segments.csv')
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='copper')
df.info()
df.head()
# resetting the format of this feature to datetime

df['attempt_date'] = pd.to_datetime(df['attempt_date'])
df.groupby('attempt_date')['attempt_date'].count().plot()
sns.scatterplot(x='act_avg_spd',y='user_age_group',data=df, hue='gender')
sns.distplot(df["act_avg_spd"])
sns.distplot(df["act_max_spd"])
sns.scatterplot(y='smt_name',x='smt_avg_spd',data=df)
df['act_max_spd_weird'] = df['act_max_spd']

df['act_max_spd_weird'] = df['act_max_spd_weird'].apply(lambda x: 1 if x > 82.52 else 0)
sns.scatterplot(y='smt_name',x='smt_avg_spd',data=df, hue='act_max_spd_weird')
sns.distplot(df["act_max_spd"]).set_xlim((20,100))
# Picking the threshold to be 50km/h from the distplot above

df['act_max_spd_weird'] = df['act_max_spd']

df['act_max_spd_weird'] = df['act_max_spd_weird'].apply(lambda x: 1 if x > 50 else 0)
sns.scatterplot(y='smt_name',x='smt_avg_spd',data=df, hue='act_max_spd_weird')