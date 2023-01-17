# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime  

import scipy

from scipy import stats



%matplotlib inline 

from sklearn import datasets

from sklearn import datasets, linear_model

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error, r2_score
# read input

INPUT_DIR = '../input'



train_df = pd.read_csv(INPUT_DIR + '/train.csv')

test_df = pd.read_csv(INPUT_DIR + '/test.csv')

y = pd.read_csv(INPUT_DIR + '/samplesubmission.csv')
# convert time convention

cols=['state_changed_at','created_at','launched_at','deadline']



for col in cols:

    train_df['%s' % col] = pd.to_datetime(train_df['%s' % col], unit='s')
# create timeline features

train_df['duration'] = (train_df.loc[:, 'deadline'] - train_df.loc[:, 'launched_at']).apply(lambda l: l.days)

train_df['month'] = train_df.loc[:, 'launched_at'].apply(lambda l: l.month)

train_df['year'] = train_df.loc[:, 'launched_at'].apply(lambda l: l.year)

train_df['day'] = train_df.loc[:, 'launched_at'].apply(lambda l: l.dayofyear)

train_df['month_end'] = train_df.loc[:, 'deadline'].apply(lambda l: l.month)

train_df['year_end'] = train_df.loc[:, 'deadline'].apply(lambda l: l.year)

train_df['day_end'] = train_df.loc[:, 'deadline'].apply(lambda l: l.dayofyear)

train_df['duration_5'] = np.ceil(train_df.loc[:, 'duration'] / 5) * 5
years_values = sorted(train_df.loc[:, 'year'].unique())

y = train_df.groupby(['year'])[['final_status']].mean()



col = train_df.groupby(['year']).count()



fig = plt.figure(figsize=(12, 5), dpi= 80)

fig.tight_layout() 

plt.subplots_adjust(wspace = .5)



ax1 = plt.subplot(1, 2, 1)

plt.title('Annual statistics')



ax2 = ax1.twinx()

ax1.bar(years_values, y['final_status'], alpha = 0.4)

ax1.set_xlabel('Years')

ax1.set_ylabel('% Funded')

ax2.set_ylabel('Total number of projects')



ax2.plot(years_values, col['goal'])



for year in years_values:

    plt.text(year-.35, 0.05, col['goal'][year], fontsize = 8)



years_values = sorted(train_df.loc[:, 'year'].unique())



plt.subplot(1, 2, 2)

#count_money = {}

train_df['money']=(train_df['goal']*train_df['final_status'])

y = train_df.groupby(['year'])['money'].sum()



plt.title('Total amount raised per year')

plt.xlabel('Years')

plt.ylabel('Funded $')



plt.bar(years_values[:-1], y[:-1], alpha = 0.4)
years_values = train_df.loc[:, 'year'].unique()

per_year_goal = {}

med={}

for year in years_values:

    per_year_goal[year] = train_df.loc[train_df['year'] == year, 'goal'].sum() / float((train_df.loc[:, 'year']==year).sum())

    goal_dist = train_df.loc[train_df['year'] == year, 'goal']

    med[year] = np.median(goal_dist)

    #df.loc[:, 'final_status']



fig = plt.figure(figsize=(12, 5), dpi= 80)

#fig.tight_layout() 

#plt.subplots_adjust(wspace = .5)



plt.subplot(1, 2, 1)

plt.bar(list(med.keys()),list(med.values()), alpha = 0.5)

plt.title('Median goal per year')

plt.ylabel('Goal ($)')



years_values = [2009, 2012, 2015]

train_df['log10_goal'] = train_df.loc[:, 'goal'].apply(lambda l: np.round(np.log10(l)))

#per_year_goal = {}

for year in years_values:

    year_filter = train_df.loc[:, 'year'] == year

    plt.subplot(3, 2, 2*np.ceil((year - 2008) / 3) )

    if year == 2012:

        plt.ylabel('# Projects')

    if year == 2015:

        plt.xlabel('Goal (log $)')

    #norm_by = len(train_df.loc[year_filter, 'goal'])

    #year_sum = train_df.groupby(['log10_goal'])[['goal']].count()

    plt.hist(train_df.loc[year_filter, 'log10_goal'], range(8))

    plt.title(str(year))

    plt.xlim(0, 7)

month_values = train_df.loc[:, 'month'].unique()

per_month_approved = {}

for month in month_values:

    per_month_approved[month] = train_df.loc[train_df['month'] == month, 'final_status'].sum() / float((train_df.loc[:, 'month']==month).sum())



plt.figure()

plt.bar(list(per_month_approved.keys()),list(per_month_approved.values()), alpha = 0.5)

plt.xlabel('Month', fontsize=12)

plt.ylabel('% Funded', fontsize=12)

plt.title('% Projects funded by launching month')
duration_values = train_df.loc[:, 'duration_5'].unique()

per_duration_approved = {}

per_duration_count = {}

for dur in duration_values:

    per_duration_approved[dur] = train_df.loc[train_df['duration_5'] == dur, 'final_status'].sum() / float((train_df.loc[:, 'duration_5']==dur).sum())

    per_duration_count[dur] = len(train_df.loc[train_df['duration_5'] == dur, 'final_status'])



    

plt.figure(figsize=(12, 5), dpi= 80)

ax = plt.subplot(1, 2, 2)

train_df['log_goal'] = train_df.loc[:, 'goal'].apply(lambda l: np.log10(l+1))

train_df.plot(kind='scatter', x='log_goal', y='duration', s = 2, alpha = 0.2, ax=ax, fontsize=10)#, colormap='plasma', c='final_status');

ax.set_xlabel('Goal (log $)', fontsize=12)

ax.set_ylabel('Duration (days)', fontsize=12)

plt.title('Goal is generally independent of duration')





plt.subplot(1, 2, 1)    

plt.bar(list(per_duration_approved.keys()),list(per_duration_approved.values()), alpha = 0.5)

plt.xlabel('Duration (days)', fontsize=12)

plt.ylabel('% Funded', fontsize=12)

plt.title('Success rate as function of duration')

f = plt.figure(figsize=(12, 5), dpi= 80)

plt.subplots_adjust(hspace = .5)

plt.subplot(2, 2, 2)

plt.hist(train_df['log_goal'], facecolor='blue', alpha = 0.25)

plt.title('Goal Distribution')

plt.xlabel('Goal (log $)')

plt.subplot(2, 2, 4)

v = train_df.loc[:, 'backers_count']+1

v[v > np.percentile(v, 99.5)] = np.percentile(v, 99.5)

plt.hist(np.log10(v), facecolor='blue', alpha = 0.25)

plt.title('Backers Distribution (not valid)')

plt.xlabel('# of backers (log backers)')

plt.subplot(1, 2, 1)

sc = plt.scatter(train_df['log_goal'], np.log10(v), c = train_df.loc[:, 'final_status'], cmap = 'Set3', alpha = 0.05)

plt.xlabel('Goal (log)', fontsize=12)

plt.ylabel('Number of backers (log)', fontsize=12)

plt.legend()

plt.xlim(0, 8.5)



status = ['Failed', 'Funded']



lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), mec="none",

                        label=status[i], ls="", marker="o")[0]

handles = [lp(i) for i in range(2)]

plt.legend(handles=handles)

plt.show()
