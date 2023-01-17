import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from __future__ import division

import pylab

plt.style.use('fivethirtyeight')

%matplotlib inline

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
data_file = "../input/2016-FCC-New-Coders-Survey-Data.csv"
df = pd.read_csv(data_file)
hist = df.Age.plot.hist(bins=75)
hist = df.Age.plot.box()
fig, ax = plt.subplots(ncols=2)

box = df.Age[df.Gender == 'male'].plot.box(ax=ax[1])

box = df.Age[df.Gender == 'female'].plot.box(ax=ax[0])

ax[1].legend(["Distribution of Male Ages"])

ax[0].legend(["Distribution of Female Ages"])
fig, ax = plt.subplots()

df[df.Gender == 'male'].Age.plot.kde(ax=ax)

plot = df[df.Gender == 'female'].Age.plot.kde(ax=ax)

legend = ax.legend(['Male', 'Female'])
fig, ax = plt.subplots()

df[df.Gender == 'male'].Age.plot.hist(bins=75, ax=ax, alpha=0.5)

df[df.Gender == 'female'].Age.plot.hist(bins=75, ax=ax, alpha=0.8)

legend = ax.legend(['Male', 'Female'])
plot = df[df.Age.notnull() == True].groupby(df.Gender).Age.size().plot.bar()

values = df[['ResourceBlogs', 'ResourceBooks', 'ResourceCodeWars','ResourceCodecademy',

'ResourceCoursera', 'ResourceDevTips', 'ResourceEdX', 'ResourceEggHead', 'ResourceFCC',

'ResourceGoogle', 'ResourceHackerRank', 'ResourceKhanAcademy', 'ResourceLynda', 'ResourceMDN',

'ResourceOdinProj', 'ResourceOther', 'ResourcePluralSight', 'ResourceReddit', 'ResourceSkillCrush',

'ResourceSoloLearn', 'ResourceStackOverflow', 'ResourceTreehouse', 'ResourceUdacity', 'ResourceUdemy',

'ResourceW3Schools', 'ResourceYouTube']].count()



bar = values.sort_values(ascending=False).plot.bar()
values = df.fillna(0)[['ResourceBlogs', 'ResourceBooks', 'ResourceCodeWars', 'ResourceCodecademy',

'ResourceCoursera', 'ResourceDevTips', 'ResourceEdX', 'ResourceEggHead', 'ResourceFCC', 'ResourceGoogle',

'ResourceHackerRank', 'ResourceKhanAcademy', 'ResourceLynda', 'ResourceMDN', 'ResourceOdinProj',

'ResourceOther', 'ResourcePluralSight', 'ResourceReddit', 'ResourceSkillCrush', 'ResourceSoloLearn',

'ResourceStackOverflow', 'ResourceTreehouse', 'ResourceUdacity', 'ResourceUdemy', 'ResourceW3Schools',

'ResourceYouTube', 'HasHighSpdInternet']].groupby('HasHighSpdInternet')

values = values.sum()



values = values.apply(lambda x: x/sum(x))

values = values.unstack().unstack()

values = values.sort_values(by=values.columns[0], axis=0)



plot = values.plot.bar(stacked=True)
values = df.fillna(0)[['ResourceBlogs', 'ResourceBooks', 'ResourceCodeWars', 'ResourceCodecademy','ResourceCoursera',

'ResourceDevTips', 'ResourceEdX', 'ResourceEggHead', 'ResourceFCC', 'ResourceGoogle', 'ResourceHackerRank','ResourceKhanAcademy',

'ResourceLynda', 'ResourceMDN', 'ResourceOdinProj', 'ResourceOther', 'ResourcePluralSight',

'ResourceReddit', 'ResourceSkillCrush', 'ResourceSoloLearn', 'ResourceStackOverflow', 'ResourceTreehouse',

'ResourceUdacity', 'ResourceUdemy', 'ResourceW3Schools','ResourceYouTube', 'Gender']].groupby('Gender')





values = values.sum()

values = values.apply(lambda x: x/sum(x))

values = values.unstack().unstack()

values = values.sort_values(by=values.columns[4], axis=0)





values.plot.bar(stacked=True)