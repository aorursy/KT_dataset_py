import seaborn as sns

from sklearn import preprocessing

import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt  

matplotlib.style.use('ggplot')

%matplotlib inline

import math

import matplotlib as mpl

import plotly

import colorsys

plt.style.use('seaborn-talk')

from mpl_toolkits.mplot3d import Axes3D

from __future__ import division

import pylab

import plotly.plotly as py

import plotly.graph_objs as go

from matplotlib import colors as mcolors

plt.style.use('fivethirtyeight')

%matplotlib inline

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

from scipy import stats

import types

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline

import plotly.plotly as py

import plotly.tools as tls





import seaborn as sns

sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()

pd.set_option('display.max_columns', 1000)

plt.rcParams['figure.figsize'] = (17.0, 6.0)

plt.rcParams['figure.titlesize'] = 16

plt.rcParams['figure.titleweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 16

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams["axes.labelsize"] = 13

plt.rcParams["axes.labelweight"] = 'bold'

plt.rcParams["xtick.labelsize"] = 12

plt.rcParams["ytick.labelsize"] = 12
df = pd.read_csv('../input/2017-fCC-New-Coders-Survey-Data.csv')

df.describe
df.head()
df.isnull().sum()
data = pd.read_csv('../input/2017-fCC-New-Coders-Survey-Data.csv')
g = sns.factorplot("Age", data=data, aspect=4, kind="count")

g.set_xticklabels(rotation=90)

g = plt.title("2017 - Distribution of New Programmers over Different Ages")
df_ageranges = df.copy()

bins=[0, 20, 30, 40, 50, 60, 100]

df_ageranges['AgeRanges'] = pd.cut(df_ageranges['Age'], bins, labels=["< 20", "20-30", "30-40", "40-50", "50-60", "< 60"]) 

df2 = pd.crosstab(df_ageranges.AgeRanges,df_ageranges.Gender).apply(lambda r: r/r.sum(), axis=1)

N = len(df_ageranges.AgeRanges.value_counts().index)

HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]

RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

ax1 = df2.plot(kind="bar", stacked=True, color= RGB_tuples, title="Gender per Age")

lines, labels = ax1.get_legend_handles_labels()

ax1.legend(lines,labels, bbox_to_anchor=(1.51, 1))
ax = sns.countplot(data=df, x='EmploymentField', hue='IsUnderEmployed')



_ = (ax.set_title('Underemployed?'),

     ax.set_xlabel('Employment Field'),

     ax.set_ylabel('Number of Coders'),

    )

plt.style.use('fivethirtyeight')

%matplotlib inline

pylab.rcParams['figure.figsize'] = (10.0, 8.0)



values = df[['CodeEventConferences', 'CodeEventDjangoGirls', 'CodeEventGameJam', 'CodeEventGirlDev', 'CodeEventHackathons', 'CodeEventMeetup', 'CodeEventNodeSchool', 'CodeEventNone', 'CodeEventOther', 'CodeEventRailsBridge', 'CodeEventRailsGirls', 'CodeEventStartUpWknd', 'CodeEventWkdBootcamps', 'CodeEventWomenCode', 'CodeEventWorkshops' ]].count()

bar = values.sort_values(ascending=False).plot.bar()

g = plt.title("Code Events")

values = data[['PodcastChangeLog', 'PodcastCodeNewbie', 'PodcastCodePen', 'PodcastDevTea', 'PodcastDotNET', 'PodcastGiantRobots', 'PodcastJSAir', 'PodcastNone', 'PodcastOther', 'PodcastProgThrowdown', 'PodcastRubyRogues', 'PodcastSEDaily', 'PodcastSERadio', 'PodcastShopTalk', 'PodcastTalkPython', 'PodcastTheWebAhead' ]].count()



bar = values.sort_values(ascending=False).plot.bar()



g = plt.title("Podcasts")
values = data[['ResourceCodecademy', 'ResourceCodeWars', 'ResourceCoursera', 'ResourceCSS', 'ResourceEdX', 'ResourceEgghead', 'ResourceHackerRank', 'ResourceKA', 'ResourceLynda', 'ResourceMDN', 'ResourceOdinProj', 'ResourceOther', 'ResourcePluralSight', 'ResourceSkillcrush', 'ResourceSO', 'ResourceTreehouse', 'ResourceUdacity', 'ResourceUdemy', 'ResourceW3S' ]].count()



bar = values.sort_values(ascending=False).plot.bar()



g = plt.title("Resources/Websites")

values = data[['YouTubeCodeCourse', 'YouTubeCodingTrain', 'YouTubeCodingTut360', 'YouTubeComputerphile', 'YouTubeDerekBanas', 'YouTubeDevTips', 'YouTubeEngineeredTruth', 'YouTubeFunFunFunction', 'YouTubeGoogleDev', 'YouTubeLearnCode', 'YouTubeLevelUpTuts', 'YouTubeMIT', 'YouTubeMozillaHacks', 'YouTubeOther', 'YouTubeSimplilearn', 'YouTubeTheNewBoston' ]].count()



bar = values.sort_values(ascending=False).plot.bar()



g = plt.title("YouTube Videos")
