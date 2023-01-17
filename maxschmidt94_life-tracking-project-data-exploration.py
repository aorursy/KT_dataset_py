import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import numpy as np



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
input_interval_file_path = '../input/life_interval_data.csv'

input_total_file_path = '../input/life_total_data.csv'



df_i = pd.read_csv(input_interval_file_path)

df_t = pd.read_csv(input_total_file_path,index_col='date')



# Parse dates and times



def conv_to_timedelta(s):

    # Hack to use pd.to_timedelta

    return pd.to_timedelta(s+':00')



def conv_to_scalar(s):

    return pd.to_timedelta(s+':00').total_seconds()/3600



def timedelta_to_scalar(td):

    return td.total_seconds()/3600



df_i.date = pd.to_datetime(df_i.date)

df_i.start_time = pd.to_datetime(df_i.start_time)



df_i.time = df_i.time.map(conv_to_timedelta)

df_i['end_time'] = df_i.start_time + df_i.time

df_i.time = df_i.time.map(timedelta_to_scalar)



df_t = df_t.applymap(conv_to_scalar)
df_i.head()
df_t.head()
df_t.describe()
df_t['productive'] = df_t['math'] + df_t['uni'] + df_t['work']

df_t.sum().plot.bar()
df_t_prod = df_t.filter(['math','uni','work'],axis=1)

df_t_prod.plot.bar(stacked=True,figsize=(16,6))
col_prod = ['math','uni','work']

df_prod = df_t[col_prod].copy()

df_prod.head()

df_prod['total'] = df_prod.apply(lambda x: x.sum(),axis=1)

df_prod.head()
ax = df_prod.total.plot.line(figsize=(8,5),title='Total productivity over time (with mean)')

ax.axhline(df_prod.total.mean(),c='grey',ls='--')
start_day = '2019-05-06'

start_loc = df_prod.index.get_loc(start_day)

prod_by_week = pd.Series()



while True:

    current_week = df_prod.total.iloc[start_loc:start_loc+7]

    if len(current_week) != 7 :

        break

    prod_by_week[df_prod.index[start_loc]] = current_week.sum()

    start_loc += 7



prod_by_week.plot.line()
df_tmp = df_prod[['total']].copy()

df_tmp['dow'] = df_prod.index.map(lambda x : pd.to_datetime(x).weekday())

df_tmp.head()
df_tmp.groupby('dow').mean().plot.bar()
ax = df_tmp.boxplot(by='dow')

ax.set_title('')

ax
work_activities = ['self','uni','work']



s_overtime = pd.Series()



for day in df_i.date.unique():

    deadline = pd.to_datetime(day)+pd.to_timedelta('20h')

    tot = 0

    over_deadline = df_i[(df_i.date==day) & (df_i.activity.isin(work_activities)) & (df_i.end_time > deadline)]

    for i,row in over_deadline.iterrows():

        if row.start_time > deadline:

            tot += row.time

        else:

            tot += timedelta_to_scalar(row.end_time - deadline)

    s_overtime[pd.to_datetime(day).strftime('%Y-%m-%d')] = tot

    

s_overtime.plot.bar(figsize=(16,6))
df_food = df_t.copy().filter(['cook','eat'],axis=1)

df_food.plot.bar(stacked=True, figsize=(16,6))
sns.distplot(df_food.cook,label='cook')

sns.distplot(df_food.eat,label='eat')



print((df_food.cook <= 0.5).sum()/len(df_food))

plt.legend()
# Assume: eating alone would take 1h every day

eat_entertainment = df_t.eat.map(lambda x: max(0, x-1))

eat_entertainment.name = 'entertainment'

sns.distplot(eat_entertainment)
all_entertainment = eat_entertainment + df_t.pause + df_t.music

print(all_entertainment.describe())

sns.distplot(all_entertainment)
df_sleep = pd.DataFrame(columns=['start','end','duration'])



days = df_t.index



for i,day in enumerate(days):



    # We only know the end time of sleep for days where we have data for the next day, too

    # So, exclude the last one



    if i == len(days)-1:

        df_sleep.loc[day] = [np.nan,np.nan,np.nan]

        continue

    

    # Get last activity of the day (which should be sleep)

    last_activity = df_i[df_i.date==day]

    if not len(last_activity):

        df_sleep.loc[day] = [np.nan,np.nan,np.nan]

        continue

    last_activity = last_activity.iloc[-1]

    

    if last_activity.activity != 'sleep':

        df_sleep.loc[day] = [np.nan,np.nan,np.nan]

        continue

    

    # Get the first activity of the next day

    first_activity = df_i[df_i.date==days[i+1]]

    if not len(first_activity):

        df_sleep.loc[day] = [np.nan,np.nan,np.nan]

        continue

    first_activity = first_activity.iloc[0]

    

    # Calculate start and end of sleep

    sleep_start = last_activity.start_time

    sleep_end = first_activity.start_time

    duration = timedelta_to_scalar(sleep_end-sleep_start)

    df_sleep.loc[day] = [sleep_start, sleep_end, duration]



plt.title('Distribution of sleep duration')

sns.distplot(df_sleep.duration.dropna())
# Nap time = all time spent sleeping except the main sleeping time



s_nap_time = df_t.sleep - df_sleep.duration

df_sleep['nap'] = s_nap_time



df_sleep.filter(['duration','nap'],axis=1).plot.bar(stacked=True,figsize=(16,5),title='Sleep split into main sleep and naps')
optimal_time = [pd.to_datetime(i)+pd.to_timedelta('22h 30min') for i in df_sleep.index]

violation = df_sleep.start - pd.Series(optimal_time,index=df_sleep.index)

violation = violation.map(timedelta_to_scalar)*60

violation.plot.bar(figsize=(16,6),color='blue')
df_tmp = df_prod[['total']].copy()

for i in range(len(df_tmp)):

    if i == 0:

        continue

    df_tmp.loc[df_tmp.index[i],'prev_sleep'] = df_sleep.iloc[i-1].duration

df_tmp.head()

df_tmp.corr()