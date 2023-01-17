import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr, spearmanr, kendalltau

sns.set(rc={'figure.figsize':(10, 8)});
df = pd.read_csv('../input/OnlineNewsPopularityReduced.csv', sep=',')
df.info()
df.describe().T
df.groupby(['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',

            'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']).size()
df['day_of_week'] = df['weekday_is_monday']

df.loc[df['weekday_is_tuesday'] == 1, 'day_of_week'] = 2

df.loc[df['weekday_is_wednesday'] == 1, 'day_of_week'] = 3

df.loc[df['weekday_is_thursday'] == 1, 'day_of_week'] = 4

df.loc[df['weekday_is_friday'] == 1, 'day_of_week'] = 5

df.loc[df['weekday_is_saturday'] == 1, 'day_of_week'] = 6

df.loc[df['weekday_is_sunday'] == 1, 'day_of_week'] = 7

df['day_of_week'].value_counts()
df.groupby('day_of_week').size().plot(kind='bar')

plt.ylabel('news')

plt.show()
df['n_tokens_title'].hist(bins = 20);
sns.boxplot(df['n_tokens_title']);
sns.jointplot(x = 'n_tokens_title', y = 'shares', data = df);
s_title_share = spearmanr(df['n_tokens_title'], df['shares'])

k_title_share = kendalltau(df['n_tokens_title'], df['shares'])

print(s_title_share, '\n', k_title_share)
s_img_share = spearmanr(df['num_imgs'], df['shares'])

print(s_img_share)

s_video_share = spearmanr(df['num_videos'], df['shares'])

print(s_video_share)
df.groupby('is_weekend')['shares'].mean().plot(kind='bar') 

plt.ylabel('shares')

plt.show();
s_content_shares = spearmanr(df['n_tokens_content'], df['shares'])

print(s_content_shares)
numeric = ['n_tokens_title', 'n_tokens_content', 'num_videos', 'num_imgs', 'num_keywords', 'shares']

sns.heatmap(df[numeric].corr(method='spearman'));
s_content_imgs = spearmanr(df['n_tokens_content'], df['num_imgs'])

print(s_content_imgs)
s_videos_imgs = spearmanr(df['num_videos'], df['num_imgs'])

print(s_videos_imgs)
s_keywords_imgs = spearmanr(df['num_keywords'], df['num_imgs'])

s_keywords_share = spearmanr(df['num_keywords'], df['shares'])

print(s_keywords_imgs, '\n', s_keywords_share)
from scipy.stats import pointbiserialr

pointbiserialr(df['shares'], df['is_weekend'])
df.groupby('day_of_week')['shares'].mean().plot(kind='bar')

plt.ylabel('shares')

plt.show();
df.groupby(['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',

            'data_channel_is_tech', 'data_channel_is_world']).size()
df.pivot_table(values=['shares'], index=['data_channel_is_lifestyle', 'data_channel_is_bus', 'data_channel_is_entertainment',

                                         'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world'], aggfunc='mean')