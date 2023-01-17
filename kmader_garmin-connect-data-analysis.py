%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
garmin_base_dir = os.path.join('..', 'input', 'garmin')
!ls -R {garmin_base_dir} | head -n 20
sleep_paths = glob(os.path.join(garmin_base_dir, 'DI_CONNECT', 'DI-Connect-Wellness', '*.json'))
print(len(sleep_paths), 'sleep files found')
def read_sleep_file(in_path):
    with open(in_path, 'r') as f:
        cur_list = json.load(f)
    cur_df = pd.DataFrame([{k: v['date'] if isinstance(v, dict) else v for k,v in  c_row.items()} for c_row in cur_list])
    for c_col in cur_df.columns:
        if ('Date' in c_col) or ('Timestamp' in c_col):
            cur_df[c_col] = pd.to_datetime(cur_df[c_col], unit='ms')
    cur_df['file_id'] = os.path.basename(in_path)
    cur_df['total_sleep_hours'] = (cur_df['sleepEndTimestampGMT']-cur_df['sleepStartTimestampGMT']).dt.total_seconds()/3600
    return cur_df
read_sleep_file(sleep_paths[-1]).sample(5)
all_sleep_df = pd.concat([read_sleep_file(x) for x in sleep_paths], sort=False).sort_values('calendarDate', ascending = False)
all_sleep_df['Year'] = all_sleep_df['calendarDate'].dt.year
all_sleep_df['DayName'] = all_sleep_df['calendarDate'].dt.day_name()
all_sleep_df['dayofweek'] = all_sleep_df['calendarDate'].dt.dayofweek
all_sleep_df.to_csv('sleep_data.csv', index = False)
all_sleep_df.head(3)
fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
ax1.plot(all_sleep_df['calendarDate'], all_sleep_df['total_sleep_hours'])
all_sleep_df['dow'] = all_sleep_df['calendarDate'].dt.dayofweek
all_sleep_df['week'] = all_sleep_df['calendarDate'].dt.week
all_sleep_df['year'] = all_sleep_df['calendarDate'].dt.year
all_sleep_df['week_idx'] = all_sleep_df['calendarDate'].dt.year*52+all_sleep_df['calendarDate'].dt.week
sleep_matrix = all_sleep_df.pivot_table(values = 'total_sleep_hours', 
                                        index = ['week', 'year'], 
                                        columns = 'dow',
                                        fill_value = np.NaN).reset_index().sort_values(['year', 'week'])
fig, ax1 = plt.subplots(1, 1, figsize = (40, 10))
sns.heatmap(sleep_matrix[list(range(7))].values.T, fmt = '2.0f', annot = True, ax = ax1)
yr_grp = sleep_matrix.groupby('year')
fig, m_axs = plt.subplots(5, 1, figsize = (40, 30))
for c_ax, (c_year, c_grp) in zip(m_axs.flatten(), yr_grp):
    sns.heatmap(c_grp.sort_values('week', ascending = True)[list(range(7))].values.T, fmt = '2.1f', annot = True, ax = c_ax)
    c_ax.set_title('Y{}'.format(c_year))
sns.pairplot(all_sleep_df[['total_sleep_hours', 'Year', 'DayName']], hue = 'DayName')
sns.pairplot(all_sleep_df[['total_sleep_hours', 'deepSleepSeconds', 'lightSleepSeconds', 'DayName']].dropna(), hue = 'DayName')
all_sleep_df[['total_sleep_hours', 'deepSleepSeconds', 'lightSleepSeconds', 'DayName']].dropna().plot.scatter('total_sleep_hours', 'deepSleepSeconds')
all_sleep_df[['total_sleep_hours', 'deepSleepSeconds', 'lightSleepSeconds', 'DayName']].dropna().plot.scatter('total_sleep_hours', 'lightSleepSeconds')
sns.factorplot(x = 'dayofweek', y = 'total_sleep_hours', hue = 'DayName', kind = 'swarm', data = all_sleep_df, size = 8)
sns.factorplot(x = 'dayofweek', y = 'deepSleepSeconds', hue = 'DayName', kind = 'swarm', data = all_sleep_df, size = 6)
sns.lmplot(x = 'total_sleep_hours', y = 'deepSleepSeconds', hue = 'DayName', data = all_sleep_df)
user_paths = glob(os.path.join(garmin_base_dir, 'DI_CONNECT', 'DI-Connect-User', 'UDS*.json'))
print(len(user_paths), 'user files found')
def read_user_file(in_path):
    with open(in_path, 'r') as f:
        cur_list = json.load(f)
    cur_df = pd.DataFrame([{k: v.get('date', '') if isinstance(v, dict) else v for k,v in  c_row.items()} for c_row in cur_list])
    
    return cur_df
all_user_df = pd.concat([read_user_file(x) for x in user_paths], sort=False).sort_values('calendarDate', ascending = False)
all_user_df['calendarDate'] = pd.to_datetime(all_user_df['calendarDate'])
all_user_df['restingHeartRateTimestamp'] =  pd.to_datetime(all_user_df['restingHeartRateTimestamp'])
all_user_df.to_csv('user_data.csv', index = False)
all_user_df.head(5)
all_user_df['activeKilocalories'].hist()
all_user_df.plot('calendarDate', 'activeKilocalories')
all_user_df.plot('calendarDate', 'restingHeartRate')
all_user_df.plot('calendarDate', 'floorsAscendedInMeters')
