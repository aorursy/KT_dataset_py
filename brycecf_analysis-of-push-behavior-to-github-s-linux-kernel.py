import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import networkx as nx

import numpy as np

import pandas as pd

import seaborn as sns

from __future__ import division
pd.set_option('display.float_format', lambda x: '%.3f' % x) #Remove scientific notation for df.describe()

commits_df = pd.read_csv('../input/LinuxCommitData.csv',

                         header=0,

                         index_col=None,

                         na_values='',

                         parse_dates=['Date'],

                         usecols=range(0,5),

                         names=['Date', 'Commits', 'Additions', 'Deletions', 'UserID'])
print(commits_df['Date'].describe())

print('='*10)

print('Average # of unique users contributing on active days: {0:.4f}'.format(22372/636))
user_total_df = commits_df.groupby('UserID').sum()

user_total_df['Net Changes'] = user_total_df['Additions'].subtract(user_total_df['Deletions'])

user_total_df['Total Changes'] = user_total_df['Additions'].add(user_total_df['Deletions'])

user_total_df.sort_values(['Commits', 'Total Changes'],

                          ascending=False,

                          inplace=True)

print('Distribution of Total User Commit Behavior (2002-2017)')

user_total_df.describe()
plt.figure(figsize=(14,2))

ax = sns.violinplot(x=user_total_df['Commits'],

                    inner='quartile',

                    saturation=0.5)

ax.tick_params(axis='both', which='major', labelsize=15)

ax.set_xlabel('# of Commits for Each One of 100 Contributors', fontsize=15)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,

                                         figsize=(14,6),

                                         sharex=True)



sns.violinplot(x=user_total_df['Total Changes'],

               inner='quartile',

               saturation=0.5,

               ax=ax1)

ax1.set_title('Total Code Changes for Each One of 100 Contributors', fontsize=15)

ax1.set_xlabel('')



sns.violinplot(x=user_total_df['Additions'],

               inner='quartile',

               saturation=0.2,

               ax=ax2,

               color='g')

ax2.set_title('Total Code Additions', fontsize=15)

ax2.set_xlabel('')



sns.violinplot(x=user_total_df['Deletions'],

               inner='quartile',

               saturation=0.4,

               ax=ax3,

               color='r')

ax3.set_title('Total Code Deletions', fontsize=15)

ax3.tick_params(axis='both', which='major', labelsize=15)

ax3.set_xlabel('')



sns.violinplot(x=user_total_df['Net Changes'],

               inner='quartile',

               saturation=0.4,

               ax=ax4,

               color='y')

ax4.set_title('Net Code Changes (Addition-Deletion)', fontsize=15)

ax4.tick_params(axis='both', which='major', labelsize=15)

ax4.set_xlabel('')
date_users_df = commits_df.groupby(['Date', 'UserID']).sum()

date_users_df['Daily Net Changes'] = date_users_df['Additions'].subtract(date_users_df['Deletions'])

date_users_df['Daily Total Changes'] = date_users_df['Additions'].add(date_users_df['Deletions'])

date_users_df.sort_values('Commits',

                          ascending=False,

                          inplace=True)

print('Distribution of Daily User Behavior')

date_users_df.describe(percentiles=[.25, .5, .6, .65, .75, .8, .9])
plt.figure(figsize=(14, 6))

ax1 = sns.distplot(date_users_df['Commits'])

ax1.tick_params(axis='both', which='major', labelsize=15)

ax1.set_title('Distribution of Daily Commits', fontsize=15)

ax1.set_xlabel('# of Commits', fontsize=15)