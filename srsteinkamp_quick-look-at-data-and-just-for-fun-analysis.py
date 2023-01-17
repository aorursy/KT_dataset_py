import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
print(os.listdir("../input"))
PATH = '../input/'
# Any results you write to the current directory are saved as output.
np.random.seed(512)
# https://nikgrozev.com/2015/06/16/fast-and-simple-sampling-in-pandas-when-loading-data-from-files/ 
# Following the approach to index the rows in pandas skiprows. Trust me with the lenght or run:
# train_lines = sum(1 for l in open(f'{PATH}train.csv'))
train_lines = 629145481 - 1 # there needs to be some substraction (0 indexing)
# if not using pandas with skiprows:
#skridx = np.arange(0, train_lines, 2)
samp_length = 75000
# I used the dtype options from aguiars kernel, as this apparently solved the memory overflow in kaggle kernel
train = pd.read_csv(f'{PATH}train.csv', dtype={'acoustic_data': np.int16, 
                                               'time_to_failure': np.float64}) # something less than 5gb, actually - things were getting worse with skiprows, so now ... 
test_ids = pd.read_csv(f'{PATH}sample_submission.csv') 

# Well decimating now here:

# skridx = np.arange(0, train_lines, 2) # Decimate and hoping not to kill the memory again. 
train = train.iloc[::2, :]
# Short overview over the file format, note everything is decimated a bit:
train.head()
# Many many samples
train.shape
# We need to predict the time_to failure for simple snippets.
test_ids.head()
# 2624 test files.
test_ids.shape 
train['time_to_failure'].describe()
train['acoustic_data'].describe()
unique_sampling_steps = np.unique(np.diff(np.round(train.time_to_failure, 9)))
# We know there are several events. So a slightly different plot:
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.hist(unique_sampling_steps)
plt.title('Steps with events')
plt.subplot(122)
plt.hist(unique_sampling_steps[unique_sampling_steps < 2])
plt.title('Removing the long events.')
plt.tight_layout()
print(np.where(np.diff(np.round(train.time_to_failure[:100000], 5)))) # I.e. decreasing precision 
# Now lets have a look at the audio signal around failures, with the index we are not really caring about being one off..
# looking for increasing times as there's a restart in the signal
failures = np.where(np.diff(train.time_to_failure) > 0)
# Seems like there are 16 failures:
print(failures[0].shape)
plt.figure(figsize=(15,10))
for ii, f_idx in zip(range(failures[0].shape[0]), failures[0]):
    plt.subplot(4, 4, ii + 1)
    plt.plot(np.arange(-samp_length, 100), train.acoustic_data[f_idx - samp_length : f_idx + 100])
plt.figure(figsize=(15,10))
for ii, f_idx in zip(range(failures[0].shape[0]), failures[0]):
    plt.subplot(4, 4, ii + 1)
    plt.plot(np.arange(-1200000, 0), train.acoustic_data[f_idx - 1200000 : f_idx])
n_rnd_samples = 1000
rnd_sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_rnd_samples)
variation = np.empty(n_rnd_samples)

for idx, samp in enumerate(rnd_sample_idx):
    variation[idx] = np.max(train.time_to_failure[samp - samp_length : samp]) - np.min(train.time_to_failure[samp - samp_length : samp])
b = plt.hist(variation[variation < 4])
b = plt.hist(np.clip(train.acoustic_data, -200, 200), bins=50)
from scipy.stats import spearmanr
spike_size = 800
spike_idx = np.abs(train.acoustic_data) > spike_size

plt.figure(figsize=(15,5))
spear = spearmanr(np.abs(train[spike_idx]['acoustic_data']), train[spike_idx]['time_to_failure'])
plt.subplot(121)
plt.scatter(np.abs(train[spike_idx]['acoustic_data']), train[spike_idx]['time_to_failure'])
b = plt.title(f'Peaks and time to failure correlate with r ={spear[0]:.3f}')
plt.subplot(122)
b = plt.hist(train[spike_idx]['time_to_failure'])
plt.tight_layout()
t_t_f = np.empty(n_rnd_samples)
m_a_d = np.empty(n_rnd_samples)

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx] = np.max(np.abs(train.acoustic_data[samp - samp_length : samp]))
spear = spearmanr(np.array(t_t_f), np.array(m_a_d))

plt.scatter(np.array(t_t_f), np.array(m_a_d))
b = plt.title(f'Peaks and time to failure correlate with r = {spear[0]:.3f}')
def basic_properties(data):
    properties = np.zeros(7)
    # For many of the basic properties we use the abs, that is the amplitude
    properties[:3] = list(np.percentile(np.abs(data), [25, 50, 75]))
    
    for n_f, jj in enumerate([np.mean, np.std, np.max, np.min]):
        properties[3 + n_f] = jj(data)
    return properties
    
t_t_f = np.empty(n_rnd_samples)
m_a_d = np.empty((n_rnd_samples, 7))

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx, :] = basic_properties(np.abs(train.acoustic_data[samp - samp_length : samp]))
plt.figure(figsize=(25,20))
feat_names = ['25-percentile', '50-percentile', '75-percentile', 'mean', 'std', 'max', 'min']
for ii, fna in enumerate(feat_names):
    ax = plt.subplot(3, 4, ii + 1)
    sns.regplot(t_t_f, m_a_d[:, ii], ax=ax)
    spear = spearmanr(np.array(t_t_f), np.array(m_a_d[:, ii]))
    b = plt.title(f'{fna} and ttf correlate with r = {spear[0]:.3f}')
t_t_f = np.empty(n_rnd_samples)
m_a_d = np.hstack([m_a_d, np.zeros((m_a_d.shape[0],1))])

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx, 7] = np.sqrt(np.mean(train.acoustic_data[samp - samp_length : samp] ** 2))
    
plt.figure(figsize=(15,5))
ax = plt.subplot(1, 2, 1)
sns.regplot(t_t_f, m_a_d[:, 7], ax=ax)
spear = spearmanr(np.array(t_t_f), np.array(m_a_d[:, 7]))
b = plt.title(f'RMS and ttf correlate with r = {spear[0]:.3f}')
ax = plt.subplot(1, 2, 2)
sns.regplot(m_a_d[:,3], m_a_d[:, 7], ax=ax)
spear = spearmanr(m_a_d[:,3], np.array(m_a_d[:, 7]))
b = plt.title(f'RMS and abs(mean) correlate with r = {spear[0]:.3f}')
# This might use up a lot of memory... 
train['acoustic_diff'] = np.hstack([0, np.diff(train['acoustic_data'])])
# simple description:
train['acoustic_diff'].describe()

def RMS(data):
    return np.sqrt(np.mean(data**2))

def basic_properties_up(data):
    properties = np.zeros(8)
    # For many of the basic properties we use the abs, that is the amplitude
    properties[:3] = np.percentile(data, [25, 50, 75])
    
    for n_f, jj in enumerate([np.mean, np.std, np.max, np.min, RMS]):
        properties[3 + n_f] = jj(data)
    return properties
    
t_t_f = np.empty(n_rnd_samples)
m_a_d = np.empty((n_rnd_samples, 8))

for idx, samp in enumerate(rnd_sample_idx):
    t_t_f[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    m_a_d[idx, :] = basic_properties_up(train.acoustic_diff[samp - samp_length : samp])
plt.figure(figsize=(25,20))
feat_names = ['25-percentile', '50-percentile', '75-percentile', 'mean', 'std', 'max', 'min', 'RMS']
for ii, fna in enumerate(feat_names):
    ax = plt.subplot(3, 4, ii + 1)
    sns.regplot(t_t_f, m_a_d[:, ii], ax=ax)
    spear = spearmanr(np.array(t_t_f), np.array(m_a_d[:, ii]))
    b = plt.title(f'{fna} and ttf correlate with r = {spear[0]:.3f}')
import librosa # Seems to be a common library for all thing auditory
# Assuming that t_t_f is in seconds, yes, I'm lazy, still didn't look that up, shame on me:
sr = round(1 / (train.iloc[0]['time_to_failure'] - train.iloc[1]['time_to_failure']))
print(sr)
(train.iloc[0]['time_to_failure'] - train.iloc[75000]['time_to_failure'])
def mfcc_wrap(data):
    return librosa.power_to_db(librosa.feature.melspectrogram(data.values.astype('float32'), n_mels=25))
plt.figure(figsize=(15,10))
for ii, f_idx in zip(range(failures[0].shape[0]), failures[0]):
    ax = plt.subplot(4, 4, ii + 1)
    sns.heatmap(mfcc_wrap(train.acoustic_data[f_idx - samp_length : f_idx + 100]))
mfcc_wrap(train.acoustic_data[ : samp_length]).ravel().shape
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
max_features = np.round(train.shape[0]/ samp_length).astype(int)
print(f'We can create {max_features} unique samples. Maybe this gets us somewhere')
def basic_properties_time(data):
    properties = np.zeros(7)
    # For many of the basic properties we use the abs, that is the amplitude
    properties[:3] = np.percentile(data, [25, 50, 75])
    
    for n_f, jj in enumerate([np.mean, np.std, np.max, RMS]):
        properties[3 + n_f] = jj(data)
    return properties
def basic_properties_diff(data):
    properties = np.zeros(6)
    properties[:2] = np.percentile(data, [25, 75])
    
    for n_f, jj in enumerate([np.min, np.std, np.max, RMS]):
        properties[2 + n_f] = jj(data)
    return properties
train =  train.drop(['acoustic_diff'], axis=1)
n_samples = 15000
X_time = np.zeros((n_samples, 7))
X_diff = np.zeros((n_samples, 6))
X_mfcc = np.zeros((n_samples, 3675))
y = np.zeros((n_samples))
sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_samples)

for idx, samp in enumerate(sample_idx):
    y[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    X_time[idx, :] = basic_properties_time(np.abs(train.acoustic_data[samp - samp_length : samp]))
    X_diff[idx, :] = basic_properties_diff(np.diff(train.acoustic_data[samp - samp_length : samp]))
    X_mfcc[idx, :] = mfcc_wrap(train.acoustic_data[samp - samp_length : samp]).ravel()
pred_mean = np.ones(y.shape) * np.mean(X_time[:, 3])
print(mean_absolute_error(y, pred_mean))
from sklearn.ensemble import RandomForestRegressor

# Make pred_tree greater than 0
pred_time = cross_val_predict(RandomForestRegressor(n_estimators=100), X_time, y, cv=3)
pred_time[pred_time < 0] = 0
print(mean_absolute_error(y, pred_time))
pred_diff = cross_val_predict(RandomForestRegressor(n_estimators=100), X_diff, y, cv=3)
pred_diff[pred_diff < 0] = 0
print(mean_absolute_error(y, pred_diff))
# simple average:
print(mean_absolute_error(y, (pred_time + pred_diff)/2))
pred_comb = cross_val_predict(RandomForestRegressor(n_estimators=100), np.hstack([X_time, X_diff]), y, cv=3)
pred_comb[pred_comb < 0] = 0
print(mean_absolute_error(y, pred_comb))
pred_mel = cross_val_predict(RandomForestRegressor(n_estimators=10, n_jobs=3), X_mfcc, y, cv=3)
pred_mel[pred_mel < 0] = 0
print(mean_absolute_error(y, pred_mel))
print(mean_absolute_error(y, (pred_mel + pred_comb)/2))
data_rms = RMS(train.acoustic_data)
print(data_rms)
n_samples = 15000
X_time = np.zeros((n_samples, 7))
X_diff = np.zeros((n_samples, 6))
X_mfcc = np.zeros((n_samples, 3675))
y = np.zeros((n_samples))
sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_samples)

for idx, samp in enumerate(sample_idx):
    y[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    temp_data = train.acoustic_data[samp - samp_length : samp]
    temp_rms = RMS(temp_data)
    temp_data = temp_data * (data_rms/temp_rms)
    
    X_time[idx, :] = basic_properties_time(np.abs(temp_data))
    X_time[idx, -1] = temp_rms # keeping the old rms as a feature basic_properties_time(np.abs(temp_data))
    X_diff[idx, :] = basic_properties_diff(np.diff(temp_data))
    X_mfcc[idx, :] = mfcc_wrap(temp_data).ravel()
pred_time = cross_val_predict(RandomForestRegressor(n_estimators=100), X_time, y, cv=3)
pred_time[pred_time < 0] = 0
print(mean_absolute_error(y, pred_time))
pred_diff = cross_val_predict(RandomForestRegressor(n_estimators=100), X_diff, y, cv=3)
pred_diff[pred_diff < 0] = 0
print(mean_absolute_error(y, pred_diff))
pred_comb = cross_val_predict(RandomForestRegressor(n_estimators=100), np.hstack([X_time, X_diff]), y, cv=3)
pred_comb[pred_comb < 0] = 0
print(mean_absolute_error(y, pred_comb))
n_samples = 15000
X_time = np.zeros((n_samples, 7))
X_diff = np.zeros((n_samples, 6))
X_mfcc = np.zeros((n_samples, 3675))
y = np.zeros((n_samples))
sample_idx = np.random.randint(low=samp_length, high=train.shape[0], size=n_samples)

for idx, samp in enumerate(sample_idx):
    y[idx] = np.median(train.time_to_failure[samp - samp_length : samp])
    X_time[idx, :] = basic_properties_time(np.abs(train.acoustic_data[samp - samp_length : samp]))
    X_diff[idx, :] = basic_properties_diff(np.diff(train.acoustic_data[samp - samp_length : samp]))
    X_mfcc[idx, :] = mfcc_wrap(train.acoustic_data[samp - samp_length : samp]).ravel()
model_basic = RandomForestRegressor(n_estimators=100)
model_basic.fit(np.hstack([X_time, X_diff]), y)
submit_df = test_ids.copy()
for seg_id in test_ids.seg_id:
    temp = pd.read_csv(f'{PATH}test/{seg_id}.csv')
    temp = temp.iloc[::2, :] # we've been using decimated data the whole time...
    x_pred = np.zeros((1, 13))
    x_pred[0, :7] = basic_properties_time(np.abs(temp.acoustic_data))
    x_pred[0, 7:] = basic_properties_diff(np.diff(temp.acoustic_data))
    X_pred = model_basic.predict(x_pred)
    
    submit_df.loc[test_ids.seg_id == seg_id, 'time_to_failure'] = np.max([0, X_pred])
submit_df.to_csv('submission.csv', index=False)