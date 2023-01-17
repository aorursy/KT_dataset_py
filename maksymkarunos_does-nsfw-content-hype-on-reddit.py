# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

reddit_data = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')

print(reddit_data.head())
reddit_data.info()
reddit_data.isnull().sum() / len(reddit_data) * 100
fig, ax = plt.subplots()

sns.countplot(reddit_data.over_18)

ax.set(xlabel='Adult content', title='Distribution of Conetent')
( reddit_data[reddit_data['over_18'] == True]['id'].count() / reddit_data.shape[0] ) * 100
fig, (ax0, ax1) = plt.subplots(

nrows=1,ncols=2, sharey=True, figsize=(14,4))

sns.kdeplot(reddit_data[reddit_data['over_18'] == True]['score'], ax=ax0, color='orange', label='Score')

sns.kdeplot(reddit_data[reddit_data['over_18'] == False]['score'], ax=ax1, color='blue', label='Score')

ax1.set(xlim=(0,1000), xlabel='Score', title='Mass-Oritnted Content')

ax0.set(xlabel='Score', title='NSFW Content', ylabel='Frequency')

ax0.axvline(x=737.9, label='Average ', linestyle='--', color='red')

ax1.axvline(x=197.2, label='Average', linestyle='--', color='red')

ax1.legend()

ax0.legend()

print(reddit_data[reddit_data['over_18'] == True]['score'].mean())

print(reddit_data[reddit_data['over_18'] == False]['score'].mean())
fig, (ax0, ax1) = plt.subplots(

nrows=1,ncols=2, sharey=True, figsize=(14,4))

sns.kdeplot(reddit_data[reddit_data['over_18'] == True]['num_comments'], ax=ax0, color='orange', label='Number of Comments')

sns.kdeplot(reddit_data[reddit_data['over_18'] == False]['num_comments'], ax=ax1, color='blue', label='Number of Comments')



ax1.set(xlim=(0,1000), xlabel='Number of Comments', title='Mass-Oritnted Content')

ax0.set(xlabel='Number of Comments', title='NSFW Content', ylabel='Frequency')

ax0.axvline(x=104.7, label='Average ', linestyle='--', color='red')

ax1.axvline(x=24.7, label='Average', linestyle='--', color='red')

ax1.legend()

ax0.legend()



print(reddit_data[reddit_data['over_18'] == True]['num_comments'].mean())

print(reddit_data[reddit_data['over_18'] == False]['num_comments'].mean())
ax = sns.regplot(x='num_comments',y='score', data=reddit_data[reddit_data['over_18'] == True])

ax.set(xlabel='Number of Comments', ylabel='Score', title='Number of Comments vs Score')
fig, ax = plt.subplots()

sns.residplot(x='num_comments',y='score', data=reddit_data[reddit_data['over_18'] == True])

ax.set(xlabel='Number of Comments', ylabel='Score', title='Residual plot')
(reddit_data[reddit_data['over_18'] == True]['removed_by'].count() / reddit_data[reddit_data['over_18'] == True]['removed_by'].ffill(0).count()) * 100
fig, (ax0, ax1) = plt.subplots(

nrows=1,ncols=2, figsize=(14,4))

sns.countplot(reddit_data[reddit_data['over_18'] == True]['removed_by'].dropna(), ax=ax0)

sns.countplot(reddit_data[reddit_data['over_18'] == False]['removed_by'].dropna(), ax=ax1)

ax0.set(xlabel='Removed by',title='NSFW content', ylim=(0,20), ylabel='Count')

ax1.set(xlabel='Removed by',title='Mass Orinted content', ylabel='Count')