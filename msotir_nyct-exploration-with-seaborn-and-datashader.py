import os.path



import datashader as ds

import datashader.transfer_functions as dtf

from datashader.colors import colormap_select as cm, inferno, viridis

from datashader.utils import lnglat_to_meters



import matplotlib as mpl

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

import seaborn as sns



from IPython.display import display, HTML

from matplotlib.colors import LinearSegmentedColormap



# Datashader helper function:

def bg(img): return dtf.set_background(img, "black")



%matplotlib inline

mpl.rcParams['figure.dpi'] = 130
def load_dataset(train_path, test_path):

    

    train = pd.read_csv(train_path)    

    test = pd.read_csv(test_path)

    

    df = pd.concat({'train': train, 'test': test}, ignore_index=False, names=['set']).reset_index(0)

        

    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)

    df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)

    

    assert len(df) == len(test) + len(train)

    

    return df
dataset_path = '/kaggle/input/' if os.path.exists('/kaggle/input/') else './data/'



df = load_dataset(train_path=os.path.join(dataset_path, 'train.csv'), test_path=os.path.join(dataset_path, 'test.csv'))
# Pickup and dropoff hour of day:

df['pickup_hour'] = df.pickup_datetime.dt.hour

df['dropoff_hour'] = df.dropoff_datetime.dt.hour



# Pickup and dropoff day of week:

df['pickup_DoW'] = df.pickup_datetime.dt.dayofweek

df['dropoff_DoW'] = df.dropoff_datetime.dt.dayofweek



# numerical trip ID:

df['id_num'] = df.id.str[2:].astype('int')



# Log trip duration (see part II):

df['log_duration'] = np.log(1 + df.trip_duration)



# convert the 'set' (train vs set) column into a categorical column (required for datashader plots):

df['set_cat'] = pd.Categorical(df.set)



# Compute trip haversine distances (geodesic distances):



def haversine(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees).

    Source: https://gis.stackexchange.com/a/56589/15183

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a)) 

    meters = 1000 * 6367 * c

    return meters



df['hdist'] = haversine(df.pickup_longitude, df.pickup_latitude,

                        df.dropoff_longitude, df.dropoff_latitude)



# Add the log haversine distance:



df['log_hdist'] = np.log(1 + df['hdist'])



# Projects longitude-latitude coordinates into Web Mercator(https://en.wikipedia.org/wiki/Web_Mercator) 

# coordinates (for visulization)



df['pickup_x'], df['pickup_y'] = lnglat_to_meters(df['pickup_longitude'], df['pickup_latitude'])

df['dropoff_x'], df['dropoff_y'] = lnglat_to_meters(df['dropoff_longitude'], df['dropoff_latitude'])
df.set_cat.value_counts()
hour_prop = df.groupby('set').pickup_hour.value_counts(dropna=False, normalize=True).reset_index(name='proportion')

sns.factorplot(x="pickup_hour", y='proportion', data=hour_prop, hue="set", size=4, aspect=2);
dow_prop = df.groupby('set').pickup_DoW.value_counts(dropna=False, normalize=True).reset_index(name='proportion')

sns.factorplot(x="pickup_DoW", y='proportion', data=dow_prop, hue="set", size=4, aspect=2);
daily_trip_counts = df.set_index('pickup_datetime').groupby('set').resample('1D').size().transpose()

daily_trip_proportion = daily_trip_counts.div(daily_trip_counts.sum())



ax = daily_trip_proportion.plot.line(figsize=(12,6));

ax.set_ylabel("daily trip proportion")
plt.figure(figsize=(12, 3))

sns.heatmap(df.groupby('set').passenger_count.value_counts(dropna=False, normalize=True).unstack(), 

            square=True, annot=True);
store_and_fwd_flag_prop = df.groupby('set').store_and_fwd_flag.value_counts(dropna=False, normalize=True).reset_index(name='proportion')

grid = sns.factorplot(x="store_and_fwd_flag", y='proportion', data=store_and_fwd_flag_prop, hue="set", size=3, aspect=0.8, kind='bar');

grid.fig.get_axes()[0].set_yscale('log')
vendor_id_prop = df.groupby('set').vendor_id.value_counts(dropna=False, normalize=True).reset_index(name='proportion')

grid = sns.factorplot(x="vendor_id", y='proportion', data=vendor_id_prop, hue="set", size=3, aspect=0.8, kind='bar');
# Bounding box where most of the data is:

nyc = {'x_range': (40.635, 40.86), 'y_range': (-74.03,-73.77)}



# Bounding box converted to Web mercator coordinates

bottom_left = lnglat_to_meters(nyc['y_range'][0], nyc['x_range'][0])

top_right = lnglat_to_meters(nyc['y_range'][1], nyc['x_range'][1])

nyc_m = {'x_range': (bottom_left[0], top_right[0]), 'y_range': (bottom_left[1], top_right[1])}
color_map = {'train':'gold', 'test':'aqua'}



# Plot train vs test set heatmap:

cvs = ds.Canvas(plot_width=1400, plot_height=1400, **nyc_m)

agg = cvs.points(df, 'pickup_x', 'pickup_y', ds.count_cat('set_cat'))

img = bg(dtf.shade(agg, color_key=color_map, how='eq_hist'))

display(img)



# Display colorbar:

fig = plt.figure(figsize=(10, 3))

fig.add_axes([0.05, 0.80, 0.9, 0.15])

cb = mpl.colorbar.ColorbarBase(ax=fig.axes[0], cmap=mpl.cm.cool.from_list('cus01', list(color_map.values())),

                               orientation='horizontal');

cb.set_ticks([0,1])

cb.set_ticklabels(list(color_map.keys()))
train = df[lambda x: x.set == 'train'].copy()
train_sample = train.sample(frac=0.15, random_state=1234)
len(train_sample)
hourly_counts = train.set_index('pickup_datetime').resample('1h').size().reset_index(name='pickups')

hourly_counts['date'] = hourly_counts.pickup_datetime.dt.strftime("%b %d %Y")

hourly_counts['hour'] = hourly_counts.pickup_datetime.dt.hour

hourly_counts['DoW'] = hourly_counts.pickup_datetime.dt.dayofweek
plt.figure(figsize=(11, 5))

sns.tsplot(time="hour", value="pickups", unit="date", condition='DoW', data=hourly_counts, err_style="unit_traces", )

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
src = train[['pickup_x', 'pickup_y']].rename(columns={'pickup_x':'x', 'pickup_y':'y'})

dst = train[['dropoff_x', 'dropoff_y']].rename(columns={'dropoff_x':'x', 'dropoff_y':'y'})



all_pts = pd.concat({'pickup': src, 'dropoff': dst}, ignore_index=False, names=['type']).reset_index(0).reset_index(drop=True)

all_pts['type'] = all_pts['type'].astype('category')



cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)

agg = cvs.points(all_pts, 'x', 'y', ds.count_cat('type'))

bg(dtf.shade(agg, color_key={'dropoff':'#FF9E25', 'pickup':'#3FBFFF'}, how='eq_hist', min_alpha=100))
train.trip_duration[lambda x: x < 3600 * 3].plot.hist(figsize=(12, 6), bins=40)
train.log_duration.plot.hist(figsize=(12, 6), bins=40)
grid = sns.jointplot(x="id_num", y="log_duration", data=train, kind='hex');

grid.fig.set_figwidth(12)

grid.fig.set_figheight(6)
grid = sns.factorplot(x="pickup_hour", y="log_duration",  hue="pickup_DoW", data=train_sample, aspect=1.5, size=8);
grid = sns.factorplot(x="passenger_count", y="trip_duration", data=train_sample, aspect=2, size=4, kind='bar');
# create plasma colormap for datashader:

plasma = [tuple(x) for x in mpl.cm.plasma(range(255), bytes=True)[:, 0:3]]
cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)

agg = cvs.points(train, 'pickup_x', 'pickup_y', ds.mean('log_duration'))

bg(dtf.shade(agg, cmap = cm(plasma, 0.2), how='eq_hist'))
cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)

agg = cvs.points(train, 'dropoff_x', 'dropoff_y', ds.mean('log_duration'))

bg(dtf.shade(agg, cmap = cm(plasma, 0.2), how='eq_hist'))
train_outliers_filtered = train[lambda x: (x.trip_duration < 3600 * 2) & (x.hdist < 30000)]



grid = sns.jointplot(x="hdist", y="trip_duration", data=train_outliers_filtered, kind='hex',

                     gridsize=80, space=0, mincnt=10, cmap='viridis')



grid.fig.set_figwidth(12)

grid.fig.set_figheight(6)
grid = sns.jointplot(x="log_hdist", y="log_duration", data=train, kind='hex', gridsize=80,

                     space=0, mincnt=10, cmap='viridis')

grid.fig.set_figwidth(12)

grid.fig.set_figheight(6)
grid = sns.jointplot(x="log_hdist", y="log_duration", data=train, kind='resid', space=0, scatter_kws={'alpha': 0.1});

grid.fig.set_figwidth(12)

grid.fig.set_figheight(6)
train['is_null_distance'] = train.hdist <= 0.05

train['is_12+_trip'] = train.trip_duration > 3600 * 12
# Create a single 'anomaly' categorical variable from ''is_null_distance' and 'is_12+_trip' flags:



train['anomaly'] = train['is_null_distance'].map({True:'null_distance', False: ''}).str.cat(

                   train['is_12+_trip'].map({True:'12+_trip', False: ''}))



train['anomaly'] = train['anomaly'].replace('', 'none').astype('category')
train['anomaly'].value_counts()
anomaly_color_map = {'none':'gray', 'null_distance':'red', '12+_trip':'yellow', 'null_distance12+_trip':'green'}



cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)



# Anomaly heatmap:

agg = cvs.points(train[lambda x:x.anomaly != 'none'], 'pickup_x', 'pickup_y', ds.count_cat('anomaly'))

img_outliers = dtf.shade(agg, color_key=anomaly_color_map, how='linear', min_alpha=140)



# Non-anomaly heatmap:

agg = cvs.points(train[lambda x:x.anomaly == 'none'], 'pickup_x', 'pickup_y', ds.count())

img_regular = dtf.shade(agg, cmap='gray', how='eq_hist', min_alpha=120)



# Show legend as HTML:

display(HTML(''.join(["<p><p style='border-left: 1.5em solid {};padding-left: 10pt;margin: 5px;'>{}</p>".format(color, label) 

              for label, color in anomaly_color_map.items()])))



# Superimpose anomaly and non-anomaly heatmaps

bg(dtf.stack(img_regular, dtf.dynspread(img_outliers)))
def anomaly_ratio(data, freq, anomaly):

    ratio = train.set_index('pickup_datetime').resample(freq)[anomaly].mean().reset_index(name='ratio')



    ratio['date'] = ratio.pickup_datetime.dt.strftime("%b %d %Y")

    ratio['hour'] = ratio.pickup_datetime.dt.hour

    ratio['weekofyear'] = ratio.pickup_datetime.dt.weekofyear

    ratio['DoW'] = ratio.pickup_datetime.dt.dayofweek

    

    return ratio
extra_long_trip_daily_ratio= anomaly_ratio(train, freq='1D', anomaly='is_12+_trip')



week_vs_DoW_extra_long_trip_ratio = extra_long_trip_daily_ratio.set_index(['DoW', 'weekofyear']).ratio.unstack()
plt.figure(figsize=(12, 3))

sns.heatmap(week_vs_DoW_extra_long_trip_ratio, square=True, cmap="summer", cbar_kws={'label':'12+ hours trip ratio'})
null_dist_trip_daily_ratio= anomaly_ratio(train, freq='1D', anomaly='is_null_distance')



week_vs_DoW_null_dist_trip_ratio = null_dist_trip_daily_ratio.set_index(['DoW', 'weekofyear']).ratio.unstack()
plt.figure(figsize=(12, 3))

sns.heatmap(week_vs_DoW_null_dist_trip_ratio, square=True, cmap="summer", cbar_kws={'label':'12+ hours trip ratio'})
hourly_anomalies_prop = (train

                         .groupby('anomaly')

                         .pickup_hour.value_counts(dropna=False, normalize=True)

                         .reset_index(name='proportion'))



# Ignore 'null_distance12+_trip' anomaly (very few points)

hourly_anomalies_prop.anomaly = hourly_anomalies_prop.anomaly.astype(str)

hourly_anomalies_prop = hourly_anomalies_prop[lambda x: x.anomaly != 'null_distance12+_trip']



sns.factorplot(x="pickup_hour", y='proportion', data=hourly_anomalies_prop, hue="anomaly", size=4, aspect=2);
def get_lines(df):

    return pd.DataFrame({

            'x': df[['dropoff_x', 'pickup_x']].assign(dummy=np.NaN).values.flatten(),

            'y': df[['dropoff_y', 'pickup_y']].assign(dummy=np.NaN).values.flatten()})
lines = get_lines(train[lambda x: x.hdist < 2000])
cvs = ds.Canvas(plot_width=2000, plot_height=2000, **nyc_m)

agg = cvs.line(lines, 'x', 'y', ds.count())

bg(dtf.shade(agg, cmap=cm(inferno, 0.1), how='log'))
import importlib



datashader_version_0_6_plus = importlib.util.find_spec("datashader.bundling") is not None
%%time



if datashader_version_0_6_plus:

    

    from datashader.bundling import directly_connect_edges, hammer_bundle # only in datashader 0.6+



    # Filter origin and destination within the region of interest: 

    train_in_nyc = train[lambda x: x.pickup_x.between(*nyc_m['x_range']) & x.pickup_y.between(*nyc_m['y_range'])

                  & x.dropoff_x.between(*nyc_m['x_range']) & x.dropoff_y.between(*nyc_m['y_range'])]



    # The `hammer bundle` function is quite slow on my machine,

    # we limit the number of displayed routes by sampling:

    train_in_nyc = train_in_nyc.sample(100000, random_state=1234)



    # Create nodes and edges dataframes:



    src = train_in_nyc[['pickup_x', 'pickup_y']].rename(columns={'pickup_x':'x', 'pickup_y':'y'})

    dst = train_in_nyc[['dropoff_x', 'dropoff_y']].rename(columns={'dropoff_x':'x', 'dropoff_y':'y'})



    nodes = pd.concat([src, dst], ignore_index=True).copy(deep=True)

    edges = pd.DataFrame({'source': list(range(len(src))), 

                          'target':list(range(len(src), len(src) + len(dst)))}).sample(frac=1).copy(deep=True)



    # Compute trip paths using the "hammer bundle" algorithm (slow!). 

    # The chosen parameter set may not be optimal, it is difficult to find

    # a good trade-off between computational performance and visual pleasantness:

    lines = hammer_bundle(nodes, edges, initial_bandwidth=0.3, decay=0.1, batch_size=20000, accuracy=300,

                          max_segment_length=0.1, min_segment_length=0.00001)
if datashader_version_0_6_plus:

    cvs = ds.Canvas(plot_width=2000, plot_height=2000)

    agg = cvs.line(lines, 'x', 'y', ds.count())

    img = bg(dtf.shade(agg, cmap=cm(inferno, 0.1), how='log'))

    display(img)

else:

    display(HTML("<img src='http://i.imgur.com/jDFJW9K.jpg'>"))