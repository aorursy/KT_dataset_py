import pandas as pd

import seaborn as sns

import matplotlib as plt

import numpy as np

%matplotlib inline

df = pd.read_csv('../input/911.csv')

df.head()
df.shape
df.describe()
# Missing values

df.isnull().sum()
# Unique twp values. I assume this stands for township?

print(len(df.twp.unique()))

df.twp.unique()
# Unique titles

print(len(df.title.unique()))

df.title.unique()
(df.e != 1).sum()
# Plot histogram of townships

sns.countplot(df.twp.values)
# Titles can be grouped by EMS, Fire and Traffic

print(df.title.map(lambda x: x.split(': ')[0]).unique())
df['call_type'] = df.title.map(lambda x: x.split(': ')[0])

df.head()
df['times'] = pd.to_datetime(df.timeStamp)

print(type(df['times'][0]))

df['Year-Month'] = df['times'].apply(lambda x: "%d-%d" % (x.year, x.month))

df['Year-Week'] = df['times'].apply(lambda x: "%d-%d" % (x.year, x.week))

#df['year'] = df['times'].apply(lambda x: "%d" % (x.year))

#df['month'] = df['times'].apply(lambda x: "%d" % (x.month))

sns.countplot(df['Year-Month'].values)
# Need to combine 2015-53 and 2016-53 - these are the same week, just broken up because it spans the year boundary.

def combine_last_week_of_2015(x):

    if x == '2016-53':

        return '2015-53'

    else:

        return x

df['Year-Week'] = df['Year-Week'].apply(combine_last_week_of_2015)

# Also, don't plot the first and last weeks, since they're incomplete.

year_week_full_intervals = df['Year-Week'][(df['Year-Week'] != '2015-50') & (df['Year-Week'] != '2016-33')]

ax = sns.countplot(year_week_full_intervals.values, color='c')

ax.set_title("Count of 911 calls by week")

ax.set_xlabel("Week")

t = ax.set_xticks(np.arange(0,34,5))
# Look at trends over the course of a week, day.

int_to_day_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

df['day_of_week'] = df['times'].apply(lambda x: int_to_day_of_week[x.dayofweek])

def plot_by_day_of_week(df, color=None, hue=None, hue_order=None):

    sns.countplot(x='day_of_week', data=df, color=color, hue=hue, hue_order=hue_order, order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

plot_by_day_of_week(df, 'y')
ems_df = df[df['call_type'] == 'EMS']

traffic_df = df[df['call_type'] == 'Traffic']

fire_df = df[df['call_type'] == 'Fire']

plot_by_day_of_week(ems_df, 'r')
plot_by_day_of_week(fire_df, 'b')
plot_by_day_of_week(traffic_df, 'g')
plot_by_day_of_week(df, color=None, hue='call_type', hue_order=['EMS', 'Traffic', 'Fire'])
df['seconds_since_midnight'] = df.times.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

ax = sns.distplot(df.seconds_since_midnight, bins=24)

def seconds_to_time_formatter(seconds, pos):

    # Add some interval and use modulo to change start time.

    # Add am/pm

    hours_since_midnight = int(seconds / 3600)

    ampm_time = hours_since_midnight

    is_pm = False

    if hours_since_midnight < 12:

        if hours_since_midnight == 0:

            ampm_time = 12

    else:

        is_pm = True

        if hours_since_midnight > 12:

            ampm_time = hours_since_midnight - 12

    return "{0}{1}".format(ampm_time, 'pm' if is_pm else 'am')

ax.xaxis.set_major_formatter(plt.ticker.FuncFormatter(seconds_to_time_formatter))

# Show a tick for every other hour

ax.xaxis.set_major_locator(plt.ticker.MultipleLocator(3600 * 2))

ax.set_xlim(0,3600*24)

ax.set_xlabel("Time of day")
# Make 4am the start of the x axis

# Translate to the right by 20 hours, then mod by 24

from scipy.stats import norm

df['seconds_since_4am'] = (df.seconds_since_midnight + 20 * 3600) % (24 * 3600)

# Graph the same way as above

ax = sns.distplot(df.seconds_since_4am, bins=48, fit=norm)

def seconds_to_time_formatter_4am(seconds, pos):

    # Add some interval and use modulo to change start time.

    # Add am/pm

    hours_since_midnight = int(seconds / 3600)

    # Adjust for 4am start time

    hours_since_midnight = (hours_since_midnight + 4) % 24

    ampm_time = hours_since_midnight

    is_pm = False

    if hours_since_midnight < 12:

        if hours_since_midnight == 0:

            ampm_time = 12

    else:

        is_pm = True

        if hours_since_midnight > 12:

            ampm_time = hours_since_midnight - 12

    return "{0}{1}".format(ampm_time, 'pm' if is_pm else 'am')

ax.xaxis.set_major_formatter(plt.ticker.FuncFormatter(seconds_to_time_formatter_4am))

ax.xaxis.set_major_locator(plt.ticker.MultipleLocator(3600 * 2))

ax.set_xlim(0,3600*24)

ax.set_xlabel("Time of day")
ax = sns.distplot(df.seconds_since_4am, bins=48, kde=False)

ax.xaxis.set_major_formatter(plt.ticker.FuncFormatter(seconds_to_time_formatter_4am))

ax.xaxis.set_major_locator(plt.ticker.MultipleLocator(3600 * 2))

ax.set_xlim(0,3600*24)

ax.set_xlabel("Time of day")

ax.set_ylabel("Calls per half hour")
g = sns.FacetGrid(df, col='call_type', size=4)

g.map(sns.distplot, 'seconds_since_4am', kde=False, bins=48)

for i in range(3):

    ax = g.facet_axis(0,i)

    ax.xaxis.set_major_formatter(plt.ticker.FuncFormatter(seconds_to_time_formatter_4am))

    ax.xaxis.set_major_locator(plt.ticker.MultipleLocator(3600 * 2))

    ax.set_xlim(0,3600*24)

    ax.set_xlabel("Time of day")

g.set_ylabels('Calls per half hour')