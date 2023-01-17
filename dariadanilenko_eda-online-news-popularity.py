# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/OnlineNewsPopularityReduced.csv')

df.head().T
weekday = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']

df2 = df[weekday].sum().T

df2.plot(kind='bar', figsize=(8, 5))

max1 = df2.idxmax()

min1 = df2.idxmin()

print("День, в который публиковалось наибольшее количество статей: " + max1)

print("День, в который публиковалось наименьшее количество статей: " + min1)
from scipy.stats import spearmanr



df['n_tokens_title'].hist(bins=20);

corr1 = spearmanr(df['n_tokens_title'], df['shares'])

print('Number of tokens in the title/shares correlation:', corr1[0], 'p-value:', corr1[1])
corr21 = spearmanr(df['num_imgs'], df['shares'])

print('Number of images/shares correlation:', corr21[0], 'p-value:', corr21[1])

corr22 = spearmanr(df['num_videos'], df['shares'])

print('Number of videos/shares correlation:', corr22[0], 'p-value:', corr22[1])
df.groupby('is_weekend')['shares'].mean().plot(kind='bar') 
corr3 = spearmanr(df['n_tokens_content'], df['shares'])

print('Number of tokens in the content/shares correlation:', corr3[0], 'p-value:', corr3[1])
theme = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']

df.pivot_table(values=['url'], index=theme, aggfunc='count')
df.pivot_table(values=['shares'], index=theme, aggfunc='mean')
themes = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world', 'shares']

sns.heatmap(df[themes].corr(method='spearman'), annot = True);
num = ['num_hrefs', 'num_keywords', 'shares']

sns.heatmap(df[num].corr(method='spearman'), annot = True);
corr4 = spearmanr(df['num_keywords'], df['shares'])

print('Number of keywords/shares correlation:', corr4[0], 'p-value:', corr4[1])