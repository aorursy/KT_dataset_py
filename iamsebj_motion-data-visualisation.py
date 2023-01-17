# Python 3 environment defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import matplotlib.dates as mdates

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
filepath = '../input/ios-core-motion-activities/motion.csv'

data = pd.read_csv(filepath, index_col='date_time', parse_dates=[['date','time']])



# Convert String 'True'/'False' columns

# Python considers 'True' and 'False' to be Booleans, so they can be used like numbers

data[['unknown','stationary','walking','running','cycling','automotive']] *= 1



# Convert 'low', 'medium', 'high' values in 'confidence' column to 0, 1, 2 respectively

#data['confidence'] = pd.Categorical(data['confidence'])

#data['confidence'] = 2 - data['confidence'].cat.codes
print("Total Row Count: {0} \nTotal Column Count: {1}".format(data.shape[0], data.shape[1]))
data.head()
data.info()
data.isnull().sum()
# Motion events which do not fit any attributes

unlabelled_data = data[(data.unknown == 0) & (data.stationary == 0) & (data.walking == 0) & (data.running == 0) & (data.cycling == 0) & (data.automotive == 0)]

print("{0} unlabelled motion events".format(unlabelled_data.shape[0]))
data_by_day = data.groupby(pd.Grouper(freq='D')).agg(len).drop(columns=['stationary','walking','running','cycling','automotive','confidence'])

data_by_day.rename(columns={'unknown':'event_count'}, inplace=True)



ax = sns.barplot(x=data_by_day.index.strftime('%d/%m'), y=data_by_day.event_count)

ax.set_title('Total Motion Events, by Day', weight='bold')

ax.set_ylabel('Motion Events Count')

ax.set_xlabel('Date');
event_count_per_hour = data.groupby(data.index.hour).count().drop(columns=['stationary','walking','running','cycling','automotive','confidence'])



_,ax = plt.subplots(figsize=(10,4))

event_count_per_hour.plot(ax=ax,legend=False)



ax.set_title('Total Motion Events, by Hour of Day', weight='bold')

ax.set_ylabel('Motion Events Count')

ax.set_xlabel('Hour of Day')

ax.xaxis.set_ticks(range(0,24));
def plot_heatmap(values_column, ax, cbar_kws=None):

    piv = pd.pivot_table(

        data=data,

        index=data.index.strftime('%d/%m'),

        columns=data.index.hour,

        values=values_column,

        aggfunc='sum',

        fill_value=0)



    sns.heatmap(

        piv,

        ax=ax,

        cmap=sns.cm.rocket_r,

        cbar_kws=cbar_kws)

    

    events_name = values_column.capitalize()

    ax.set_title(f'{events_name} Events', weight='bold')

    ax.set_xlabel('Hour')

    ax.set_ylabel('Day')

    ax.collections[0].colorbar.set_label(f'Count')



_,ax = plt.subplots(2, 2, figsize=(14,8))

plt.subplots_adjust(hspace=0.4)



plot_heatmap('walking', ax[0, 0])

plot_heatmap('automotive', ax[0, 1])

plot_heatmap('running', ax[1, 0], cbar_kws={"ticks":[0,1,2]})

plot_heatmap('cycling', ax[1, 1], cbar_kws={"ticks":[0,1,2]})
confidence_data = data.confidence.value_counts()



_,ax = plt.subplots(figsize=(8,6))

confidence_data.plot(kind='pie', ax=ax, autopct='%.0f%%')

ax.set_title('Confidence', weight='bold')

ax.set_ylabel(None);
data.loc[data.confidence == 'medium'].drop(columns=['confidence']).sum()