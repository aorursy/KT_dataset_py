import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import chain
sns.set_style('whitegrid')
warnings.simplefilter('ignore', FutureWarning)
train_series = pd.read_csv('../input/training_set.csv')
train_metadata = pd.read_csv('../input/training_set_metadata.csv')
train_series.head(10)
ts_lens = train_series.groupby(['object_id', 'passband']).size()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(ts_lens, ax=ax)
ax.set_title('distribution of time series lengths')
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train_series['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train_series[train_series['object_id'] == 615]['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(train_series[(train_series['object_id'] == 615) 
                          & (train_series['passband'] == 2)]['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()
obj_obs_count = train_series['object_id'].value_counts().reset_index()
obj_obs_count.columns = ['object_id', 'count']

obj_obs_count_w_ddf = pd.merge(
    obj_obs_count, train_metadata[['object_id', 'ddf']], on='object_id')

selected = obj_obs_count_w_ddf.groupby('ddf')['count'].value_counts()
selected.index.names = ['ddf', 'count_val']
selected = selected.reset_index().pivot('count_val', 'ddf',
                                        'count').rename(columns={
                                            0: 'nonddf',
                                            1: 'ddf'
                                        }).fillna(0)
selected['total'] = selected['nonddf'] + selected['ddf']

f, ax = plt.subplots(figsize=(12, 6))
ax.vlines(x=selected.index, ymin=0, ymax=selected['total'], 
          colors='red', label='ddf')
ax.vlines(x=selected.index, ymin=0, ymax=selected['nonddf'], 
          colors='blue', label='non-ddf')
ax.set_title('number of observations per object')
plt.legend()
plt.show()
simple_features = train_series.groupby(
    ['object_id', 'passband'])['flux'].agg(
    ['mean', 'max', 'min', 'std']).unstack('passband')
simple_features.head().T
print(f'mjd unique values: {train_series["mjd"].nunique()}')
print(f'int (1day) mjd unique values: {train_series["mjd"].astype(int).nunique()}')
print(f'''int mjd bins: {train_series["mjd"].astype(int).max()
      - train_series["mjd"].astype(int).min() + 1}''')
print(f'5day mjd unique values: {(train_series["mjd"]/5).astype(int).nunique()}')
print(f'''5day mjd bins: {int((train_series["mjd"].astype(int).max()
      - train_series["mjd"].astype(int).min()) / 5 + 1)}''')
# binning
ts_mod = train_series[['object_id', 'mjd', 'passband', 'flux']].copy()
ts_mod['mjd_d5'] = (ts_mod['mjd'] / 5).astype(int)
ts_mod = ts_mod.groupby(['object_id', 'mjd_d5', 'passband'])['flux'].mean().reset_index()
ts_mod.head(10)
# pivotting
ts_piv = pd.pivot_table(ts_mod, 
                        index='object_id', 
                        columns=['mjd_d5', 'passband'], 
                        values='flux',
                        dropna=False)
ts_piv.head(10)
# resetting column index to fill mjd_d5 gaps 
t_min, t_max = ts_piv.columns.levels[0].min(), ts_piv.columns.levels[0].max()
t_range = range(t_min, t_max + 1)
mux = pd.MultiIndex.from_product([list(t_range), list(range(6))], 
                                 names=['mjd_d5', 'passband'])
ts_piv = ts_piv.reindex(columns=mux).stack('passband')
ts_piv.head(10)
np.mean(np.ravel(pd.isna(ts_piv).values))
def time_kernel(diff, tau):
    return np.exp(-diff ** 2 / (2 * tau ** 2))
t_min, t_max = train_series['mjd'].min(), train_series['mjd'].max()
t_min, t_max
sample_points = np.array(np.arange(t_min, t_max, 20))
sample_points
weights = time_kernel(np.expand_dims(sample_points, 0) 
                      - np.expand_dims(train_series['mjd'].values, 1), 5)
ts_mod = train_series[['object_id', 'mjd', 'passband', 'flux']].copy()
for i in range(len(sample_points)):
    ts_mod[f'sw_{i}'] = weights[:, i]
def group_transform(chunk):
    sample_weights = chunk[[f'sw_{i}' for i in range(len(sample_points))]]
    sample_weights /= np.sum(sample_weights, axis=0)
    weighted_flux = np.expand_dims(chunk['flux'].values, 1) * sample_weights.fillna(0)
    return np.sum(weighted_flux, axis=0)
# only using a small sample as this step is slower than the other steps
ts_samp = ts_mod[ts_mod['object_id'].isin([615, 713])].groupby(
    ['object_id', 'passband']).apply(group_transform)
ts_samp
groups = train_series.groupby(['object_id', 'passband'])
def normalise(ts):
    return (ts - ts.mean()) / ts.std()
times = groups.apply(
    lambda block: block['mjd'].values).reset_index().rename(columns={0: 'seq'})
flux = groups.apply(
    lambda block: normalise(block['flux']).values
).reset_index().rename(columns={0: 'seq'})
times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
import cesium.featurize as featurize
from scipy import signal
import warnings
# def extract_freq(t, m, e):
#     fs = np.linspace(2*np.pi/0.1, 2*np.pi/500, 10000)
#     pgram = signal.lombscargle(t, m, fs, normalize=True)
#     return fs[np.argmax(pgram)]

# N = 20
# warnings.simplefilter('ignore', RuntimeWarning)
# feats = featurize.featurize_time_series(times=times_list[:N],
#                                         values=flux_list[:N],
#                                         features_to_use=['freq1'],
#                                         custom_functions={'freq1': extract_freq},
#                                         scheduler=None)
warnings.simplefilter('ignore', RuntimeWarning)
N = 100
cfeats = featurize.featurize_time_series(times=times_list[:N],
                                        values=flux_list[:N],
                                        features_to_use=['freq1_freq',
                                                        'freq1_signif',
                                                        'freq1_amplitude1'],
                                        scheduler=None)
cfeats.stack('channel').iloc[:24]
def plot_phase(n, fr):
    selected_times = times_list[n]
    selected_flux = flux_list[n]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    f, ax = plt.subplots(figsize=(12, 6))
    for band in range(6):
        ax.scatter(x=(selected_times[band] * fr) % 1, 
                   y=selected_flux[band], 
                   c=colors[band])
    ax.set_xlabel('phase')
    ax.set_ylabel('relative flux')
    ax.set_title(
        f'object {train_metadata["object_id"][n]}, class {train_metadata["target"][n]}')
    plt.show()
plot_phase(0, 3.081631)
plot_phase(3, 0.001921)
plot_phase(6, 1.005547)
plot_phase(46, 1.781711)
plot_phase(62, 4.771011)
plot_phase(69, 1.597659)
plot_phase(91, 2.668811)
nonddf_pos = train_metadata[train_metadata['ddf'] == 0].index
nonddf_times_list = [v for i, v in enumerate(times_list) if i in set(nonddf_pos)]
nonddf_flux_list = [v for i, v in enumerate(flux_list) if i in set(nonddf_pos)]
warnings.simplefilter('ignore', RuntimeWarning)
N = 50
cfeats = featurize.featurize_time_series(times=nonddf_times_list[:N],
                                        values=nonddf_flux_list[:N],
                                        features_to_use=['freq1_freq',
                                                        'freq1_signif',
                                                        'freq1_amplitude1'],
                                        scheduler=None)
cfeats.stack('channel').iloc[:24]
plot_phase(nonddf_pos[15], 4.440506)
plot_phase(nonddf_pos[20], 0.831637)
plot_phase(nonddf_pos[39], 0.817696)
plot_phase(nonddf_pos[46], 2.514430)