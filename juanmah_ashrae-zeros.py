import numpy as np

import pandas as pd

import plotly.express as px

from IPython.core.display import display, HTML

from ashrae_utils import reduce_mem_usage
data_path = '../input/ashrae-energy-prediction/'
X_train = pd.read_csv(data_path + 'train.csv', engine='python')

X_train['building_id'] = pd.Categorical(X_train['building_id'])

X_train['timestamp'] = pd.to_datetime(X_train['timestamp'], format='%Y-%m-%d %H:%M:%S')

X_train['meter'] = pd.Categorical(X_train['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
X_train.head()
n_samples = X_train.shape[0]

n_samples_zero = X_train[X_train['meter_reading']==0.0].shape[0]

display(HTML(f'''There are {n_samples_zero:,} zeros for a total of {n_samples:,} samples.<br>

The ratio of zeros is: {n_samples_zero/n_samples:.2%}.'''))
X_train_building =  X_train[['building_id', 'meter_reading']]

zeros_building = X_train_building.groupby(['building_id']).agg(['count', np.count_nonzero])

zeros_building['meter_reading', 'count_nonzero'] = zeros_building['meter_reading', 'count_nonzero'].astype(int)

zeros_building['meter_reading', 'count_zero'] = zeros_building['meter_reading']['count'] - zeros_building['meter_reading']['count_nonzero']

zeros_building.columns = zeros_building.columns.droplevel(0)

zeros_building.rename_axis(None, axis=1)

zeros_building.sort_values(by=['count_zero'], ascending=False, inplace=True)

zeros_building.reset_index(inplace=True)

zeros_building.to_csv('zeros_building.csv', index = False)

zeros_building.head()
n_buildings = zeros_building.shape[0]

n_buildings_nonzero = zeros_building[zeros_building['count_zero'] == 0].shape[0]

display(HTML(f'''There are {n_buildings_nonzero} buildings without any zero value

for a total of {n_buildings} buildings.<br>

The ratio of buildings without any zero value is: {n_buildings_nonzero/n_buildings:.2%}.'''))
fig = px.bar(zeros_building[zeros_building['count_zero'] > 0],

             x='building_id',

             y='count_zero')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
X_train_building_aspect =  X_train[['building_id', 'meter', 'meter_reading']]

zeros_building_aspect = X_train_building_aspect.groupby(['building_id', 'meter']).agg(['count', np.count_nonzero])

zeros_building_aspect['meter_reading', 'count_zero'] = zeros_building_aspect['meter_reading']['count'] - zeros_building_aspect['meter_reading']['count_nonzero']

zeros_building_aspect.columns = zeros_building_aspect.columns.droplevel(0)

zeros_building_aspect.rename_axis(None, axis=1)

zeros_building_aspect.sort_values(by=['count_zero'], ascending=False, inplace=True)

zeros_building_aspect.reset_index(inplace=True)

zeros_building_aspect.to_csv('zeros_building_aspect.csv', index = False)

zeros_building_aspect.head()
# The groupby created all combinations of building and meter, even if them doesn't exist.

# Then we get only the existing combinations.

zeros_building_aspect_notna = zeros_building_aspect[zeros_building_aspect['count_zero'].notna()]

zeros_building_aspect_notzero = zeros_building_aspect_notna[zeros_building_aspect_notna['count_zero']==0]

n_series = zeros_building_aspect_notna.shape[0]

n_series_notzero = zeros_building_aspect_notzero.shape[0]

display(HTML(f'''There are {n_series_notzero} series without any zero value

for a total of {n_series} series.<br>

The ratio of series without any zero value is: {n_series_notzero/n_series:.2%}.'''))
zbann = zeros_building_aspect_notna[['meter', 'building_id']].groupby('meter').count()

zbann.columns = ['total']

zbanz = zeros_building_aspect_notzero[['meter', 'building_id']].groupby('meter').count()

zbanz.columns = ['without zero values']

timeseries_aspect = zbanz.join(zbann)

timeseries_aspect['with zero values'] = timeseries_aspect['total'] - timeseries_aspect['without zero values']

timeseries_aspect['wozv ratio'] = round(timeseries_aspect['without zero values'] / timeseries_aspect['total'] * 100, 2).astype('str') + ' %'

timeseries_aspect['wzv ratio'] = round(timeseries_aspect['with zero values'] / timeseries_aspect['total'] * 100, 2).astype('str') + ' %'

timeseries_aspect[['without zero values', 'wozv ratio', 'with zero values', 'wzv ratio', 'total']]
zeros_building_aspect_zero = zeros_building_aspect_notna[zeros_building_aspect_notna['count_zero']>0].copy()

zeros_building_aspect_zero['building_id-meter'] = zeros_building_aspect_zero['building_id'].astype('str') + '-' + zeros_building_aspect_zero['meter']

fig = px.bar(zeros_building_aspect_zero,

             x='building_id-meter',

             y='count_zero')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
fig = px.bar(zeros_building_aspect_zero,

             x='building_id-meter',

             y='count_zero',

             color='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
fig = px.histogram(zeros_building_aspect_zero,

                   x='count_zero',

                   facet_col='meter',

                   histnorm='percent',

                   nbins=10)

fig.show()
z = zeros_building_aspect_zero['count_zero'].value_counts().iloc[0:10]



display(HTML(f'''Repeated number of zeros:<br><br>

<pre><code>{z}</code></pre>'''))
cases = [3377, 46, 206, 7, 218, 3378]



def zeros_aspect(df, zeros):

    za = df[df['count_zero'] == zeros][['meter', 'building_id']].groupby('meter').count()

    za.columns = [zeros]

    display(za)



for case in cases:

    zeros_aspect(zeros_building_aspect_zero, case)
X_train_aspect =  X_train[['meter', 'meter_reading']]

zeros_aspect = X_train_aspect.groupby(['meter']).agg(['count', np.count_nonzero])

zeros_aspect['meter_reading', 'count_nonzero'] = zeros_aspect['meter_reading', 'count_nonzero'].astype(int)

zeros_aspect['meter_reading', 'count_zero'] = zeros_aspect['meter_reading']['count'] - zeros_aspect['meter_reading']['count_nonzero']

zeros_aspect.columns = zeros_aspect.columns.droplevel(0)

zeros_aspect.rename_axis(None, axis=1)

zeros_aspect = zeros_aspect.reset_index()

zeros_aspect['percentage'] = round(zeros_aspect['count_zero'] / zeros_aspect['count'] * 100, 2).astype('str') + ' %'

zeros_aspect
X_train_hour =  X_train[['timestamp', 'meter', 'meter_reading']]

X_train_hour['hour'] = X_train_hour['timestamp'].dt.hour

zeros_hour = X_train_hour.groupby(['hour', 'meter']).agg(['count', np.count_nonzero])

zeros_hour['meter_reading', 'count_nonzero'] = zeros_hour['meter_reading', 'count_nonzero'].astype(int)

zeros_hour['meter_reading', 'count_zero'] = zeros_hour['meter_reading']['count'] - zeros_hour['meter_reading']['count_nonzero']

zeros_hour.columns = zeros_hour.columns.droplevel(0)

zeros_hour.rename_axis(None, axis=1)

zeros_hour = zeros_hour.reset_index()

zeros_hour.head()
fig = px.bar(zeros_hour,

             x='hour',

             y='count_zero',

             facet_row='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
X_train_dayofweek =  X_train[['timestamp', 'meter', 'meter_reading']]

X_train_dayofweek['dayofweek'] = X_train_dayofweek['timestamp'].dt.dayofweek

zeros_dayofweek = X_train_dayofweek.groupby(['dayofweek', 'meter']).agg(['count', np.count_nonzero])

zeros_dayofweek['meter_reading', 'count_nonzero'] = zeros_dayofweek['meter_reading', 'count_nonzero'].astype(int)

zeros_dayofweek['meter_reading', 'count_zero'] = zeros_dayofweek['meter_reading']['count'] - zeros_dayofweek['meter_reading']['count_nonzero']

zeros_dayofweek.columns = zeros_dayofweek.columns.droplevel(0)

zeros_dayofweek.rename_axis(None, axis=1)

zeros_dayofweek = zeros_dayofweek.reset_index()

zeros_dayofweek.head()
fig = px.bar(zeros_dayofweek,

             x='dayofweek',

             y='count_zero',

             facet_row='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
X_train_month = X_train[['timestamp', 'meter', 'meter_reading']]

X_train_month['month'] = X_train_month['timestamp'].dt.month

zeros_month = X_train_month.groupby(['month', 'meter']).agg(['count', np.count_nonzero])

zeros_month['meter_reading', 'count_nonzero'] = zeros_month['meter_reading', 'count_nonzero'].astype(int)

zeros_month['meter_reading', 'count_zero'] = zeros_month['meter_reading']['count'] - zeros_month['meter_reading']['count_nonzero']

zeros_month.columns = zeros_month.columns.droplevel(0)

zeros_month.rename_axis(None, axis=1)

zeros_month = zeros_month.reset_index()

zeros_month.head()
fig = px.bar(zeros_month,

             x='month',

             y='count_zero',

             facet_row='meter')

fig.update_layout(xaxis={'type': 'category'})

fig.show()
def correlativeness(series):

    

    # Initialize variables

    count = 0

    total_zeros = 0

    local_zeros = 0

    clusters = []

    

    # Iterate series

    for i in series:

        # Count

        count += 1

        # If the element is zero, increment both total and local zeros counter

        if i == 0:

            total_zeros += 1

            local_zeros += 1

        else:

            # If the last iteration was a zero, then append local zeros counter to clusters

            # and reset local zeros counter

            if local_zeros > 0:

                clusters.append(local_zeros)

                local_zeros = 0

                

    # Append local zeros counter to clusters in case it is not saved before

    if local_zeros > 0:

        clusters.append(local_zeros)

        

    # Calculate mean and standard deviation

    if len(clusters) > 0:

        mean = np.mean(clusters)

        std = np.std(clusters) / mean

    else:

        mean = np.NaN

        std = np.NaN



    return count, total_zeros, len(clusters), mean, std, clusters
X_train_correlative = X_train[['building_id', 'meter', 'meter_reading']]

correlative = X_train_correlative.groupby(['building_id', 'meter']).agg([correlativeness]).dropna()

correlative_ext = pd.DataFrame(correlative[('meter_reading', 'correlativeness')].values.tolist(), index=correlative.index)

correlative[['count', 'zeros', 'clusters', 'mean', 'std', 'detail']] = correlative_ext

correlative.drop(['meter_reading'], axis='columns', inplace=True)

correlative.columns = correlative.columns.droplevel(1)

correlative.reset_index(inplace=True)

correlative.head()
# Plotly has no logarithmic x axis in histogram

import matplotlib.pyplot as plt

import seaborn as sns
g = sns.FacetGrid(correlative, col='meter')

g = g.set(xscale='log')

g = g.map(plt.hist, 'zeros', bins=np.logspace(0, np.log10(24*366), num=25).astype('int'))
g = sns.FacetGrid(correlative, col='meter')

g = g.set(xscale='log')

g = g.map(plt.hist, 'clusters',bins=np.logspace(0, 3, num=25).astype('int'))
g = sns.FacetGrid(correlative, col='meter')

g = g.set(xscale='log')

g = g.map(plt.hist, 'mean', bins=np.logspace(0, np.log10(24*366), num=25))
g = sns.FacetGrid(correlative, col='meter')

g = g.set(xscale='log')

g = g.map(plt.hist, 'std', bins=np.logspace(0, 1, num=25))
def count_zeros(series):

    zeros = 0

    for i in series:

        if i == 0:

            zeros += 1

    return zeros
X_train_dayofyear = X_train[['timestamp', 'building_id', 'meter', 'meter_reading']].copy()

X_train_dayofyear['dayofyear'] = X_train_dayofyear['timestamp'].dt.dayofyear

X_train_dayofyear.drop('timestamp', axis='columns', inplace=True)

zeros_dayofyear = X_train_dayofyear.groupby(['dayofyear', 'building_id', 'meter']).agg([count_zeros]).dropna()

zeros_dayofyear = zeros_dayofyear.reset_index()

zeros_dayofyear.columns = zeros_dayofyear.columns.droplevel(1)

zeros_dayofyear.rename(columns={'meter_reading':'count_zeros'}, inplace=True)

zeros_dayofyear.head()
fig = px.density_heatmap(zeros_dayofyear[zeros_dayofyear['meter'] == 'electricity'],

                         x='dayofyear',

                         y='building_id',

                         z='count_zeros',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category'})

fig.show()
fig = px.density_heatmap(zeros_dayofyear[zeros_dayofyear['meter'] == 'chilledwater'],

                         x='dayofyear',

                         y='building_id',

                         z='count_zeros',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category'})

fig.show()
fig = px.density_heatmap(zeros_dayofyear[zeros_dayofyear['meter'] == 'steam'],

                         x='dayofyear',

                         y='building_id',

                         z='count_zeros',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category'})

fig.show()
fig = px.density_heatmap(zeros_dayofyear[zeros_dayofyear['meter'] == 'hotwater'],

                         x='dayofyear',

                         y='building_id',

                         z='count_zeros',

                         histfunc='sum',

                         nbinsx=366,

                         nbinsy=1449,

                         height=1600)

fig.update_layout(xaxis={'type': 'category'})

fig.show()