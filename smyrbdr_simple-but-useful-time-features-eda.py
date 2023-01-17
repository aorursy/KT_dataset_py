import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
print(os.listdir("../input"))
PATH_TO_DATA = ('../input')
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), index_col='session_id')
times = ['time%s' % i for i in range(1, 11)]

train_df[times] = train_df[times].apply(pd.to_datetime)
# Create a separate dataframe where we will work with timestamps
time_df = pd.DataFrame(index=train_df.index)

time_df['min'] = train_df[times].min(axis=1) 
time_df['max'] = train_df[times].max(axis=1)
time_df['minutes'] = round(((time_df['max'] - time_df['min']) / np.timedelta64(1, 's'))/60,2)


time_df['start_month'] = time_df['min'].apply(lambda ts: 100 * ts.year + ts.month)
time_df['year'] = time_df['min'].apply(lambda ts: ts.year)
time_df['month'] = time_df['min'].apply(lambda ts: ts.month)

time_df['start_week'] = time_df['min'].apply(lambda ts: 100 * ts.year + ts.week)
time_df['start_day'] = time_df['min'].apply(lambda ts: ts.timetuple().tm_yday)
time_df['start_hour'] = time_df['min'].apply(lambda ts: ts.hour)

time_df['dow'] = time_df['min'].apply(lambda ts: ts.date().weekday())
time_df['is_weekend'] = time_df['min'].apply(lambda ts: 1 if ts.date().weekday() in (5, 6) else 0)
time_df['work_hours'] = time_df['min'].apply(lambda ts: 1 if (ts.date().weekday() in (0,1,2,3)) &
                                             ((ts.hour>=8)&(ts.hour<=17)&(ts.hour!=12)) else 0)
time_df['target'] = train_df['target']
time_df.head()
plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
sns.countplot(x="dow", hue="target", data=time_df)
plt.subplot(1, 3, 2)
plt.title('Alice day of weeks')
sns.countplot(x="dow", data=time_df[time_df['target']==1])
plt.subplot(1, 3, 3)
plt.title('Intruder day of weeks')
sns.countplot(x="dow", data=time_df[time_df['target']==0])
plt.tight_layout()
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.title('Alice % activity at a day of week')
ax =sns.barplot(x="dow", y="dow", data=time_df[time_df['target']==1], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==1]) * 100)
ax.set(ylabel="Percent")

plt.subplot(1, 2, 2)
plt.title('Intruder % activity at a day of week')
ax =sns.barplot(x="dow", y="dow", data=time_df[time_df['target']==0], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==0]) * 100)
ax.set(ylabel="Percent")
plt.tight_layout()
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.title('Alice start hour percentage')
ax =sns.barplot(x="start_hour", y="start_hour", data=time_df[time_df['target']==1], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==1]) * 100)
ax.set(ylabel="Percent")

plt.subplot(1, 2, 2)
plt.title('Intruder start hour percentage')
ax =sns.barplot(x="start_hour", y="start_hour", data=time_df[time_df['target']==0], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==0]) * 100)
ax.set(ylabel="Percent")
plt.tight_layout()
plt.figure(figsize=(20, 8))

plt.title('Alice day of years')
sns.countplot(x="start_day", data=time_df[time_df['target']==1])
plt.tight_layout()
sns.heatmap(time_df.corr())
sns.lmplot('minutes', 'start_hour', data=time_df, hue='target', fit_reg=False)
sns.lmplot('minutes', 'dow', data=time_df, hue='target', fit_reg=False)
plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
plt.title('Alice % activity in 2013 and in 2014')
ax =sns.barplot(x="year", y="year", data=time_df[time_df['target']==1], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==1]) * 100)
ax.set(ylabel="Percent")
plt.subplot(1,2,2)
plt.title('Intruder % activity in 2013 and in 2014')
ax =sns.barplot(x="year", y="year", data=time_df[time_df['target']==0], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==0]) * 100)
ax.set(ylabel="Percent")
plt.tight_layout()
plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
plt.title('Alice monthly % activity')
ax =sns.barplot(x="month", y="month", data=time_df[time_df['target']==1], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==1]) * 100)
ax.set(ylabel="Percent")
plt.subplot(1,2,2)
plt.title('Intruder monthly % activity')
ax =sns.barplot(x="month", y="month", data=time_df[time_df['target']==0], 
                 estimator=lambda x: len(x) / len(time_df[time_df['target']==0]) * 100)
ax.set(ylabel="Percent")
plt.tight_layout()