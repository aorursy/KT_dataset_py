# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.style.use('ggplot')

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mpl.rcParams['axes.titlesize'] = 20

mpl.rcParams['axes.labelsize'] = 16
def reduce_mem_usage(df):

    """

    Reduce dataframe's memory usage

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    numerics = ['int8', 'int16', 'int32', 'int64', 'float16',

                'float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64']



    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int' or str(col_type)[:4] == 'uint':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min >= np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:

                    df[col] = df[col].astype(np.uint8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min >= np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:

                    df[col] = df[col].astype(np.uint16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min >= np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:

                    df[col] = df[col].astype(np.uint32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

                elif c_min >= np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:

                    df[col] = df[col].astype(np.uint64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimisation is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(

        100 * (start_mem - end_mem) / start_mem))



    return df
def corr_plot(data, title, method='pearson', figsize=(13,8)):

    """

    Plot the correlation matrix

    """

    mname = {

        'pearson': 'Pearson correlation',

        'kendall': 'Kendall Tau correlation',

        'spearman': 'Spearman rank correlation'

    }

    corr = data.corr(method=method)

    fig, (ax) = plt.subplots(1, 1, figsize=figsize)

    ax.set_title("{} ({})".format(title, mname[method]))

    ax = sns.heatmap(

        corr,

        vmin=-1, vmax=1, center=0,

        cmap=sns.diverging_palette(20, 220, n=200),

        square=True

    )

    ax.set_yticklabels(

        ax.get_yticklabels(),

        rotation=0,

        horizontalalignment='right'

    )

    ax.set_xticklabels(

        ax.get_xticklabels(),

        rotation=90,

        horizontalalignment='right'

    )

    ax.set_ylim(corr.shape[0], 0)

    return fig, (ax)
%%time

df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv").pipe(reduce_mem_usage)
# Lowercase all columns

df.columns = map(str.lower, df.columns)
df.describe(include='O').T
df.describe().T
%%time

df = df.assign(

    start_time=lambda df: pd.to_datetime(df.start_time),

    end_time=lambda df: pd.to_datetime(df.end_time),

    weather_timestamp=lambda df: pd.to_datetime(df.weather_timestamp),

    time_span=lambda df: df.end_time - df.start_time,

    time_span_hour=lambda df: df.time_span / np.timedelta64(1, 'h'),

    time_span_minute=lambda df: df.time_span_hour * 60,

    start_hour=lambda df: df.start_time.dt.hour,

    start_month=lambda df: df.start_time.dt.month,

    start_dow=lambda df: df.start_time.dt.weekday_name,

    start_dom=lambda df: df.start_time.dt.day,

    end_hour=lambda df: df.end_time.dt.hour,

    end_month=lambda df: df.end_time.dt.month,

    end_dow=lambda df: df.end_time.dt.dayofweek,

    end_dom=lambda df: df.end_time.dt.day,

)
_ = corr_plot(df.drop(['start_time', 'end_time', 'time_span', 'weather_timestamp'], axis=1), title='Correlation plot', method='pearson', figsize=(15, 10))
plt.figure(figsize=(12,8))

ax = df['source'].value_counts().plot(kind='bar')

ax.set_title("Report count by source")

ax.set_xlabel("Source")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(17,8))

ax = df['state'].value_counts().plot(kind='bar')

ax.set_title("Accident count by state")

ax.set_xlabel("State")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(17,8))

ax = df['state'].value_counts().sort_values(ascending=True).tail(10).plot(kind='barh')

ax.set_title("Accidents by state - Top 10 states with most accidents")

ax.set_ylabel("State")

ax.get_xaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(17,8))

ax = df['state'].value_counts().sort_values(ascending=True).head(10).plot(kind='barh')

ax.set_title("Accidents by state - Top 10 states with the least number of accidents")

ax.set_ylabel("State")

ax.get_xaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(17,8))

ax = df['severity'].value_counts().plot(kind='bar')

ax.set_title("Accident count by severity")

ax.set_xlabel("Severity")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
df.groupby('state').agg(

    accident_count=('id','count'),

    mean_severity=('severity', 'mean'),

    median_severity=('severity', 'median'),

    std_severity=('severity', 'std')

).sort_values(by=['mean_severity', 'accident_count'], ascending=[False, False]).head(10)
plt.figure(figsize=(14,8))

ax = df.plot(kind='scatter', x='start_lng', y='start_lat', label='Severity', c='severity', cmap=plt.get_cmap('jet'), colorbar=True,alpha=0.4, figsize=(14,8))

ax.set_title("Severity distribution by location")

# ax.set_xlabel("Severity")

ax.legend()

plt.ioff()
plt.figure(figsize=(17,8))

ax = df['start_hour'].value_counts().sort_index().plot(kind='bar')

ax.set_title("Accident starts at which hour?")

ax.set_xlabel("Hour")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(17,8))

ax = df['start_dow'].value_counts().plot(kind='bar')

ax.set_title("Accident starts at which day of week?")

ax.set_xlabel("Hour")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
df.time_span_hour.describe()
df[df.time_span_hour < 0].id.count()
df = df[(df['time_span_hour'] > 0)]
df.query("time_span_hour < 24")['id'].count() / df.id.count()
plt.figure(figsize=(14,8))

# ax = df.query("time_span_hour <= 24")['time_span_hour'].hist(bins=50)

ax = sns.kdeplot(df.query("time_span_hour <= 24")['time_span_hour'], shade=True)

ax.set_title("Accident length (in hour) - Accidents that happened within 24 hours")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(14,8))

# ax = df.query("time_span_hour <= 24")['time_span_hour'].hist(bins=50)

ax = sns.kdeplot(df.query("time_span_hour <= 2")['time_span_minute'], shade=True)

ax.set_title("Accident length (in minute) - Accidents that happened within 2 hours")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(13,8))

ax = sns.boxplot(x='severity', y='time_span_hour', data=df[df['time_span_hour'] <= 24])

ax.set_title("Duration distribution by severity")
plt.figure(figsize=(13,8))

ax = df['timezone'].value_counts().plot(kind='bar')

ax.set_title("Accident count by timezone")

ax.set_xlabel("Timezone")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
df['distance(mi)'] = df['distance(mi)'].astype('float')
df['distance(mi)'].describe()
df['distance(mi)'].quantile(.99)
plt.figure(figsize=(14,8))

# ax = df.query("time_span_hour <= 24")['time_span_hour'].hist(bins=50)

ax = sns.kdeplot(df[df['distance(mi)'] < 5]['distance(mi)'], shade=True)

ax.set_title("Length of road affected (in mile)")

ax.get_yaxis().set_major_formatter(

    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))

)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)
plt.figure(figsize=(13,8))

ax = sns.boxplot(x='severity', y='distance(mi)', data=df[df['distance(mi)'] < 5])

ax.set_title("Distance distribution by severity")