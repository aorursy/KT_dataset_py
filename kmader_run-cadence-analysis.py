%matplotlib inline

import os, sys

base_data_dir = os.path.join('..', 'input')

sys.path.append(os.path.join(base_data_dir, 'fitparse', 'python-fitparse-master'))

from glob import glob

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import librosa

import fitparse

plt.style.use('ggplot')
experiments_list = {os.path.dirname(x) for x in glob(os.path.join(base_data_dir, '*', '*.fit'))}

experiments_list
test_exp = list(experiments_list)[-2]

cur_fit = glob(os.path.join(test_exp, '*.fit'))[0]

cur_csv = glob(os.path.join(test_exp, '*.csv'))[0]

cur_aud = glob(os.path.join(test_exp, '*.wav'))
fit_data = fitparse.FitFile(cur_fit)

fit_df = pd.DataFrame([

    {k['name']: k['value']

     for k in a.as_dict()['fields']} 

    for a in fit_data.get_messages('record')])

fit_df['elapsed_time'] = (fit_df['timestamp']-fit_df['timestamp'].min()).dt.total_seconds()

fit_df.sample(3)
if 'vertical_ratio' not in fit_df.columns:

    fit_df['vertical_ratio'] = fit_df['vertical_oscillation']/100

if 'step_length' not in fit_df.columns:

    fit_df['step_length'] = fit_df['speed']*1000/60/fit_df['cadence']

fit_df.describe()
fit_df.plot(x='elapsed_time', y='altitude')

fit_df.plot(x='elapsed_time', y='heart_rate')
sl_df = pd.read_csv(cur_csv)

sl_df = 0.5*(sl_df.fillna(method='backfill')+sl_df.fillna(method='ffill'))

sl_df = sl_df.fillna(method='backfill').fillna(method='ffill')

sl_df['elapsed_time'] = sl_df['relative_time']/1000

sl_df.plot(x='elapsed_time', y='AccY')

sl_df.sample(5)
fig, ax1 = plt.subplots(1, 1)

ax1.hist(1/np.diff(sl_df['elapsed_time']), 

         np.linspace(990, 1010, 50))

ax1.set_xlabel('Frequency (hz)')

ax1.set_ylabel('Count');
time_start = 500

time_length = 30

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (30, 10))

df_filt = lambda x: x.query(f'elapsed_time>{time_start}').query(f'elapsed_time<={time_start+time_length}')

imu_wind_df = df_filt(sl_df)

fit_wind_df = df_filt(fit_df)

ax1.plot(imu_wind_df['elapsed_time'], imu_wind_df['AccX'], label='X')

ax1.plot(imu_wind_df['elapsed_time'], imu_wind_df['AccY'], label='Y')

ax1.plot(imu_wind_df['elapsed_time'], imu_wind_df['AccZ'], label='Z')

ax1.legend()

ax1.set_title('IMU')

ax2.plot(fit_wind_df['elapsed_time'], fit_wind_df['cadence'])

ax2.set_title('Cadence')

ax3.plot(fit_wind_df['elapsed_time'], fit_wind_df['vertical_oscillation'])

ax3.set_title('Vertical Oscillation')
from scipy.signal import spectrogram

from scipy.interpolate import interp1d
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

f, t, Sxx = spectrogram(sl_df['AccY'], fs=1000)

ax1.pcolormesh(t, f, np.sqrt(Sxx), cmap='viridis')

ax1.set_ylabel('Frequency [Hz]')

ax1.set_xlabel('Time [sec]')

ax1.set_title('Spectrogram')

ax2.plot(t, f[np.argmax(Sxx, 0)])

ax2.set_title('Maximum Frequency')
new_sample_freq = 20

new_imu_time = np.arange(sl_df['elapsed_time'].min(), sl_df['elapsed_time'].max(), 1/new_sample_freq)

it_func = lambda col_name: interp1d(sl_df['elapsed_time'], sl_df[col_name], kind='linear')(new_imu_time)

new_imu_df = pd.DataFrame({'elapsed_time': new_imu_time, 

                          'AccX': it_func('AccX'),

                          'AccY': it_func('AccY'),

                          'AccZ': it_func('AccZ'),

                          })

print(new_imu_df.shape, sl_df.shape)

new_imu_df.head(5)
time_length = 30

np.random.seed(2017)

fig, m_axs = plt.subplots(3, 3, figsize = (30, 10))

for (ax1, ax2, ax3) in m_axs.T:

    time_start = np.random.uniform(0, new_imu_df['elapsed_time'].max()-time_length)

    df_filt = lambda x: x.query(f'elapsed_time>{time_start}').query(f'elapsed_time<={time_start+time_length}')

    imu_wind_df = df_filt(new_imu_df)

    fit_wind_df = df_filt(fit_df)

    ax1.plot(imu_wind_df['elapsed_time'], imu_wind_df['AccX'], label='X')

    ax1.plot(imu_wind_df['elapsed_time'], imu_wind_df['AccY'], label='Y')

    ax1.plot(imu_wind_df['elapsed_time'], imu_wind_df['AccZ'], label='Z')

    ax1.legend()

    ax1.set_title(f'IMU: {time_start/60:2.1f} minutes')

    ax2.plot(fit_wind_df['elapsed_time'], fit_wind_df['cadence'])

    ax2.set_title('Cadence')

    ax3.plot(fit_wind_df['elapsed_time'], fit_wind_df['vertical_ratio'])

    ax3.set_title('Vertical Ratio')
fig, m_axs = plt.subplots(3, 3, figsize=(20, 20))

spectral_dict = {}

for axis_name, (ax1, ax2, ax3) in zip('XYZ', m_axs):

    f, t, Sxx = spectrogram(new_imu_df[f'Acc{axis_name}'], fs=60*new_sample_freq)

    ax1.pcolormesh(t, f, np.log10(Sxx), cmap='viridis')

    ax1.set_ylabel('Frequency [Hz]')

    ax1.set_xlabel('Time [sec]')

    ax1.set_title(f'Spectrogram {axis_name}')

    ax2.plot(t, f[np.argmax(Sxx, 0)])

    ax2.set_title('Maximum Frequency')

    ax3.plot(t, np.max(Sxx, 0))

    ax3.set_title('Maximum Amplitude')

    spectral_dict['elapsed_time'] = t*60

    spectral_dict[f'Acc{axis_name}_Frequency'] = f[np.argmax(Sxx, 0)]

    spectral_dict[f'Acc{axis_name}_Amplitude'] = np.max(Sxx, 0)

spectral_df = pd.DataFrame(spectral_dict)

spectral_df.sample(3)
fit_df.columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

ax1.plot(fit_df['elapsed_time'], fit_df['cadence'])

ax1.set_title('Fitness Tracker Data\nCadence')

ax2.plot(spectral_df['elapsed_time'], spectral_df['AccX_Frequency'])

ax2.set_title('Belt-Pack IMU Data\nX Maximum Frequency')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

ax1.plot(fit_df['elapsed_time'], fit_df['stance_time'])

ax1.set_title('Fitness Tracker Data\nStance Time')

ax2.plot(spectral_df['elapsed_time'], spectral_df['AccY_Amplitude'])

ax2.set_title('Belt-Pack IMU Data\n$Y$ Maximum Ampitude')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

ax1.plot(fit_df['elapsed_time'], fit_df['vertical_oscillation'])

ax1.set_title('Fitness Tracker Data\nVertical Oscillation')

ax2.plot(spectral_df['elapsed_time'], spectral_df['AccY_Amplitude'])

ax2.set_title('Belt-Pack IMU Data\n$Z$ Maximum Ampitude')

ax2.set_ylim(0, 20)