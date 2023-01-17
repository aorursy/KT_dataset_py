# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.

import os
# Any results you write to the current directory are saved as output.

reg_df = pd.read_csv('../input/RegularSeasonCompactResults.csv')
print('There are %s games in this dataset' % reg_df.shape[0])

reg_df['Wmod10'] = reg_df.WScore % 10
reg_df['Lmod10'] = reg_df.LScore % 10
reg_df['score'] = list(zip(reg_df['Wmod10'], reg_df['Lmod10']))
counts = reg_df['score'].value_counts()
percentages = counts / reg_df.shape[0]
print(percentages.head(5))

w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('Regular Season Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()
tourney_df = pd.read_csv('../input/NCAATourneyCompactResults.csv')
print('There are %s games in this dataset' % tourney_df.shape[0])
tourney_df['Wmod10'] = tourney_df.WScore % 10
tourney_df['Lmod10'] = tourney_df.LScore % 10
tourney_df['score'] = list(zip(tourney_df['Wmod10'], tourney_df['Lmod10']))
counts = tourney_df['score'].value_counts()
percentages = counts / tourney_df.shape[0]
print(percentages.head(5))
w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('NCAA Tournament Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()
secondary_df = pd.read_csv('../input/SecondaryTourneyCompactResults.csv')
print('There are %s games in this dataset' % secondary_df.shape[0])
secondary_df['Wmod10'] = secondary_df.WScore % 10
secondary_df['Lmod10'] = secondary_df.LScore % 10
secondary_df['score'] = list(zip(secondary_df['Wmod10'], secondary_df['Lmod10']))
counts = secondary_df['score'].value_counts()
percentages = counts / secondary_df.shape[0]
print(percentages.head(5))
w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('Secondary Tournament Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()

total_df = pd.concat([reg_df, tourney_df, secondary_df])
counts = total_df['score'].value_counts()
percentages = counts / total_df.shape[0]
w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('Combined Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()
counts = total_df['score'].value_counts()
percentages = counts / total_df.shape[0]
print(percentages.head(10))