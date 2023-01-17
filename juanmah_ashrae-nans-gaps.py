import numpy as np

import pandas as pd

import plotly.express as px

from IPython.core.display import display, HTML

import pickle

from datetime import timedelta, date

from tqdm.notebook import tqdm
data_path = '../input/ashrae-data-wrangling-csv-to-pickle/'

with open(data_path + 'X_train.pickle', 'rb') as f:

    X_train = pickle.load(f)

with open(data_path + 'X_test.pickle', 'rb') as f:

    X_test = pickle.load(f)

with open(data_path + 'nan.pickle', 'rb') as f:

    nan = pickle.load(f)
X_train_gap = X_train[['building_id', 'meter', 'meter_reading']]

gap = X_train_gap.groupby(['building_id', 'meter']).agg(['count'])

gap['meter_reading', 'gap'] = 24*366 - gap['meter_reading']['count']

gap.columns = gap.columns.droplevel(0)

gap.rename_axis(None, axis=1)

gap.sort_values(by=['gap'], ascending=False, inplace=True)

gap.reset_index(inplace=True)

gap.to_csv('train_gap.csv', index = False)

gap.head()
n_samples = X_train.shape[0]

n_samples_gap = gap['gap'].sum()

display(HTML(f'''In train set, there are {n_samples_gap:,} gaps for a total of {(n_samples + n_samples_gap):,} samples.<br>

The ratio of gaps is: {n_samples_gap / (n_samples + n_samples_gap):.2%}.'''))
X_test_gap = X_test[['building_id', 'meter', 'row_id']]

test_gap = X_test_gap.groupby(['building_id', 'meter']).agg(['count'])

test_gap['row_id', 'gap'] = 24*365*2 - test_gap['row_id']['count']

test_gap.columns = test_gap.columns.droplevel(0)

test_gap.rename_axis(None, axis=1)

test_gap.sort_values(by=['gap'], ascending=False, inplace=True)

test_gap.reset_index(inplace=True)

test_gap.to_csv('test_gap.csv', index = False)

test_gap.sample(5)
n_samples = X_test.shape[0]

n_samples_gap = test_gap['gap'].sum()

display(HTML(f'''In test set, there are {n_samples_gap:,} gaps for a total of {n_samples:,} samples.<br>

The ratio of gaps is: {n_samples_gap/n_samples:.2%}.'''))
theoretical_maximum_samples_in_test_set = 2380*24*365*2

display(HTML(f'''Theoretical maximum of samples in test set: {theoretical_maximum_samples_in_test_set:,} samples<br>

2380 timesets × 24 hours × 365 days × 2 years'''))
n_timeseries = gap.shape[0]

n_timeseries_gap = gap[gap['gap']==0].shape[0]

display(HTML(f'''There are {n_timeseries_gap} timeseries without any gap

for a total of {n_timeseries} timeseries.<br>

The ratio of timeseries without any gap is: {n_timeseries_gap/n_timeseries:.2%}.'''))
nan_building_aspect = nan.groupby(['building_id', 'meter']).agg(['sum'])

nan_building_aspect.columns = nan_building_aspect.columns.droplevel(1)

nan_building_aspect.rename(columns={"isNaN": "count_NaN"}, inplace=True)

nan_building_aspect.sort_values(by=['count_NaN'], ascending=False, inplace=True)

nan_building_aspect.reset_index(inplace=True)

display(HTML(f'''Top 5 timeseries with more NaNs:'''))

nan_building_aspect.head()
nan_building_aspect_graph = nan_building_aspect[nan_building_aspect['count_NaN']>0].copy()

nan_building_aspect_graph['building_id-meter'] = nan_building_aspect_graph['building_id'].astype('str') + '-' + nan_building_aspect_graph['meter'].astype('str')

fig = px.bar(nan_building_aspect_graph,

             x='building_id-meter',

             y='count_NaN')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
n_samples = nan_building_aspect_graph['count_NaN'].sum()

display(HTML(f'''There are {n_samples} samples with NaN values.

This value matches with the missing values count.'''))
fig = px.bar(nan_building_aspect_graph,

             x='building_id-meter',

             y='count_NaN',

             color='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
fig = px.histogram(nan_building_aspect_graph,

                   x='count_NaN',

                   facet_row='meter',

                   histnorm='percent',

                   range_x=[0,2000],

                   nbins=800)

fig.show()
gap_building = gap.groupby(['building_id']).agg(['sum'])

gap_building.sample(5)
n_buildings = gap_building.shape[0]

n_buildings_gap = gap_building[gap_building['gap']['sum']==0].shape[0]

display(HTML(f'''There are {n_buildings_gap} buildings without any gap

for a total of {n_buildings} buildings.<br>

The ratio of buildings without any gap is: {n_buildings_gap/n_buildings:.2%}.'''))
nan_building = nan_building_aspect.groupby(['building_id']).agg(['sum'])

nan_building.columns = nan_building.columns.droplevel(1)

nan_building.sort_values(by=['count_NaN'], ascending=False, inplace=True)

nan_building.reset_index(inplace=True)

nan_building.head()
nan_building_graph = nan_building[nan_building['count_NaN']>0].copy()

fig = px.bar(nan_building_graph,

             x='building_id',

             y='count_NaN')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
n_buildings = nan_building[nan_building['count_NaN']==0]['building_id'].count()

display(HTML(f'''There are {n_buildings} buildings without any gap.

This value matches with the previous approximation.'''))
z = nan_building_aspect['count_NaN'].value_counts().iloc[0:10]



display(HTML(f'''Repeated number of gaps:<br><br>

<pre><code>{z}</code></pre>'''))
def get_timestamps(df):

    ts = pd.DataFrame(columns=['building_id', 'meter', 'timestamp'])

    for index, row in df.iterrows():

        timestamps = nan[(nan['building_id']==row['building_id']) &\

                         (nan['meter']==row['meter']) &\

                         (nan['isNaN']==True)]['timestamp']

        for timestamp in timestamps:

            ts = ts.append({'building_id': row['building_id'],

                             'meter': row['meter'],

                             'timestamp': timestamp}, ignore_index=True)

    return ts
one_nan = get_timestamps(nan_building_aspect[nan_building_aspect['count_NaN']==1])

one_nan_value_counts = one_nan['timestamp'].value_counts()

one_nan_value_counts.name = 'n_timestamps'

display(HTML(f'''Most repeated timestamps for timeseries which contains only one NaN:<br>

{pd.DataFrame(one_nan_value_counts[one_nan_value_counts > 10]).to_html()}'''))
two_nans = get_timestamps(nan_building_aspect[nan_building_aspect['count_NaN']==2])

two_nans_value_counts = two_nans['timestamp'].value_counts()

two_nans_value_counts.name = 'n_timestamps'

display(HTML(f'''Most repeated timestamps for timeseries which contains exactly two NaNs:<br>

{pd.DataFrame(two_nans_value_counts[two_nans_value_counts > 10]).to_html()}'''))
three_nans = get_timestamps(nan_building_aspect[nan_building_aspect['count_NaN']==3])

three_nans_value_counts = three_nans['timestamp'].value_counts()

three_nans_value_counts.name = 'n_timestamps'

display(HTML(f'''Most repeated timestamps for timeseries which contains exactly three NaNs:<br>

{pd.DataFrame(three_nans_value_counts[three_nans_value_counts > 10]).to_html()}'''))
cases = [1312, 1452]



def nans_aspect(df, nans):

    za = df[df['count_NaN'] == nans][['meter', 'building_id']].groupby('meter').count()

    za.columns = [nans]

    display(za)



for case in cases:

    nans_aspect(nan_building_aspect, case)
X_train_aspect = X_train[['meter', 'meter_reading']].groupby(['meter']).agg(['count'])

X_train_aspect.columns = X_train_aspect.columns.droplevel(0)

X_train_aspect

nan_aspect = nan_building_aspect.groupby(['meter']).agg(['sum'])

nan_aspect.columns = nan_aspect.columns.droplevel(1)

nan_aspect['count_NaN'] = nan_aspect['count_NaN'].astype(int)

nan_aspect = pd.merge(X_train_aspect, nan_aspect, on='meter')

nan_aspect.reset_index(inplace=True)

nan_aspect['percentage'] = round(nan_aspect['count_NaN'] / nan_aspect['count'] * 100, 2).astype('str') + ' %'

nan_aspect
nan_hour =  nan[['timestamp', 'meter', 'isNaN']]

nan_hour['hour'] = nan_hour['timestamp'].dt.hour

nan_hour = nan_hour.groupby(['hour', 'meter']).agg([np.count_nonzero])

nan_hour.drop('timestamp', axis='columns', inplace=True)

nan_hour.columns = nan_hour.columns.droplevel(0)

nan_hour.rename(columns={'count_nonzero': 'count_NaN'}, inplace=True)

nan_hour.reset_index(inplace=True)

nan_hour.pivot(index='meter', columns='hour', values='count_NaN')
fig = px.bar(nan_hour,

             x='hour',

             y='count_NaN',

             facet_row='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
nan_dayofweek =  nan[['timestamp', 'meter', 'isNaN']]

nan_dayofweek['dayofweek'] = nan_dayofweek['timestamp'].dt.dayofweek

nan_dayofweek = nan_dayofweek.groupby(['dayofweek', 'meter']).agg(['count', np.count_nonzero])

nan_dayofweek.drop('timestamp', axis='columns', inplace=True)

nan_dayofweek.columns = nan_dayofweek.columns.droplevel(0)

nan_dayofweek.rename(columns={'count_nonzero': 'count_NaN'}, inplace=True)

nan_dayofweek.reset_index(inplace=True)

nan_dayofweek.pivot(index='meter', columns='dayofweek', values='count_NaN')
fig = px.bar(nan_dayofweek,

             x='dayofweek',

             y='count_NaN',

             facet_row='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
nan_month =  nan[['timestamp', 'meter', 'isNaN']]

nan_month['month'] = nan_month['timestamp'].dt.month

nan_month = nan_month.groupby(['month', 'meter']).agg(['count', np.count_nonzero])

nan_month.drop('timestamp', axis='columns', inplace=True)

nan_month.columns = nan_month.columns.droplevel(0)

nan_month.rename(columns={'count_nonzero': 'count_NaN'}, inplace=True)

nan_month.reset_index(inplace=True)

nan_month.pivot(index='meter', columns='month', values='count_NaN')
fig = px.bar(nan_month,

             x='month',

             y='count_NaN',

             facet_row='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
nan_dayofyear =  nan[['timestamp', 'building_id', 'meter', 'isNaN']]

nan_dayofyear['dayofyear'] = nan_dayofyear['timestamp'].dt.dayofyear

nan_dayofyear = nan_dayofyear.groupby(['dayofyear', 'building_id', 'meter']).agg(['count', np.count_nonzero])

nan_dayofyear.drop('timestamp', axis='columns', inplace=True)

nan_dayofyear.columns = nan_dayofyear.columns.droplevel(0)

nan_dayofyear.rename(columns={'count_nonzero': 'count_NaN'}, inplace=True)

nan_dayofyear.reset_index(inplace=True)

nan_dayofyear.head()
fig = px.density_heatmap(nan_dayofyear[nan_dayofyear['meter'] == 'electricity'],

                         x='dayofyear',

                         y='building_id',

                         z='count_NaN',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category',

                         'tickformat': '%d %B %Y'})

fig.show()
fig = px.density_heatmap(nan_dayofyear[nan_dayofyear['meter'] == 'chilledwater'],

                         x='dayofyear',

                         y='building_id',

                         z='count_NaN',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category',

                         'tickformat': '%d %B %Y'})

fig.show()
fig = px.density_heatmap(nan_dayofyear[nan_dayofyear['meter'] == 'steam'],

                         x='dayofyear',

                         y='building_id',

                         z='count_NaN',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category',

                         'tickformat': '%d %B %Y'})

fig.show()
fig = px.density_heatmap(nan_dayofyear[nan_dayofyear['meter'] == 'hotwater'],

                         x='dayofyear',

                         y='building_id',

                         z='count_NaN',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category',

                         'tickformat': '%d %B %Y'})

fig.show()