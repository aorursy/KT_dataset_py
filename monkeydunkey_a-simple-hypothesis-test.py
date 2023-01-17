import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
runs = pd.read_csv("../input/runs.csv")

runs.head(10)
grp_age = runs[['horse_age', 'time1', 'time2', 'time3']].groupby('horse_age').mean()

grp_age['mean_time'] = (grp_age['time1'] + grp_age['time2'] + grp_age['time3'])/3

grp_age[['mean_time']].plot.bar()
uni_horse_age = runs[['horse_age', 'horse_id']].groupby('horse_age').horse_id.nunique()

uni_horse_age.plot.bar()
runs = runs[runs.horse_age == 3][['horse_country', 'horse_type', 'horse_gear', 'declared_weight', 'actual_weight',\

                                  'time1', 'time2', 'time3', 'horse_age']]
runs['mean_time'] = (runs['time1'] + runs['time2'] + runs['time3']) / 3

sns.heatmap(runs[['mean_time', 'declared_weight', 'actual_weight']].corr())
runs[['mean_time', 'declared_weight', 'actual_weight']].corr()
# horse_country', 'horse_type', 'horse_gear'

grp_country = runs[['horse_country','horse_type', 'mean_time']].dropna().groupby(['horse_country','horse_type']).mean()

grp_country[['mean_time']].sort_values('mean_time').head(15).plot.bar()
# Lets see how many observation points we have

runs[(runs.horse_country == 'SAF') & (runs.horse_type == 'Mare')]
runs.mean_time.describe()
runs.mean_time.std()
runs[(runs.horse_country == 'SAF') & (runs.horse_type == 'Mare')].mean_time.describe()