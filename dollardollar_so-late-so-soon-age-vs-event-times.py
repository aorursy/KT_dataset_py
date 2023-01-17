import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pickle

import seaborn as sns

%matplotlib inline



from scipy import sparse

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,FunctionTransformer



#path to data and features

DATA_PATH = "../input/"
events = pd.read_csv('{0}events.csv'.format(DATA_PATH)).loc[:, ['timestamp', 'device_id']]

events['timestamp'] = pd.to_datetime(events['timestamp'])
def fract_hour(time):

    return time.hour + time.minute / 60.0 + time.second / 3600.0

events['hour'] = events['timestamp'].apply(lambda time: fract_hour(time))
events['hour_recentred'] = [((time + 2) % 24)-2 for time in events['hour']]
ax1 = sns.distplot(events['hour_recentred'])



ax1.set_xlim(xmin = -2, xmax = 22)

ax1.set_xlabel('Hour of day')

ax1.set_title('Events by hour -- recentered')
age_sex = pd.read_csv('{0}gender_age_train.csv'.format(DATA_PATH)).drop('group', axis = 1)

age_sex_event = age_sex.merge(events, 'inner', on = 'device_id').drop_duplicates().drop('device_id', axis = 1)
age_sex_event['bin'] = pd.cut(age_sex_event['hour_recentred'], [-2, 2, 7, 22])



ax = sns.violinplot(x="bin", y="age", data = age_sex_event)

ax.set_ylim(ymin = 18, ymax = 55)

ax.set_xlabel('Time of day')

ax.set_title('Age distribution by time of day')
ax_violin = sns.violinplot(x='bin', y='age', hue = 'gender', split = False, data = age_sex_event)



ax_violin.set_ylim(ymin = 18, ymax = 55)

ax_violin.set_xlabel('Time of day')

ax_violin.set_title('Age distribution by time of day and gender')

ax_violin.legend(bbox_to_anchor=(1.05, 1), loc=2)