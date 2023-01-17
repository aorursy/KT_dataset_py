!pip install -qq fitparse
%load_ext autoreload

%autoreload 2

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import os

import fitparse

base_dir = os.path.join('..', 'input')
all_df = pd.read_csv(os.path.join(base_dir, 'activities.csv'))

all_df['filename'] = all_df['filename'].map(str).map(lambda x: os.path.join(base_dir, x.replace('.gz', '')))

all_df['file_ext'] = all_df['filename'].map(lambda x: os.path.splitext(x)[1][1:])

all_df = all_df.query('file_ext=="fit"').sort_values('date', ascending=True)

print(all_df.shape)

all_df.tail(8)
fit_keys = {'altitude': 'ele', 'timestamp': 'time', 'heart_rate': 'hr'}

def read_fit(in_path):

    fit_data = fitparse.FitFile(in_path)

    fit_df = pd.DataFrame([

        {fit_keys.get(k['name'], k['name']): k['value']

         for k in a.as_dict()['fields']} 

        for a in fit_data.get_messages('record')])

    fit_df['time'] = pd.to_datetime(fit_df['time'])

    fit_df['elapsed_time'] = (fit_df['time']-fit_df['time'].min()).dt.total_seconds()

    return fit_df

sample_fit_df = read_fit(all_df['filename'].iloc[-6])

sample_fit_df.sample(5)
fig, m_axs = plt.subplots(5, 5, figsize=(25, 25))

for c_ax, c_col in zip(m_axs.flatten(), sample_fit_df.select_dtypes(['float', 'int']).columns):

    sample_fit_df[c_col].hist(ax=c_ax)

    c_ax.set_title(c_col)
from tqdm import tqdm_notebook

ride_list = [read_fit(c_row['filename']).assign(activity_id=c_row['id'])  for _, c_row in tqdm_notebook(list(

    all_df.query('type=="Ride"').tail(30).iterrows()))

            ]

all_power_df = pd.concat([x for x in ride_list if 'power' in x.columns])

all_power_df.reset_index(drop=True, inplace=True)

all_power_df.sample(3)
all_power_df.shape, sample_fit_df.shape
# reformat and keep only data with HR, Power and Cadence

power_df = all_power_df.sort_values(['activity_id', 'elapsed_time']).dropna(subset=['hr', 'power', 'cadence']).copy()

power_df['cum_time'] = np.cumsum(np.clip(power_df['elapsed_time'].diff(), 0, 5)) # no more than a 5 second gap between metrics

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

ax1.plot(power_df['cum_time'], power_df['power'], '.')

ax1.plot(power_df['cum_time'], power_df['power'].rolling(100, center=True).mean(), '-', label='smoothed')

ax1.legend()

ax1.set_title('Power')



ax2.plot(power_df['cum_time'], power_df['hr'], '.')

ax2.plot(power_df['cum_time'], power_df['hr'].rolling(100, center=True).mean(), '-', label='smoothed')

ax2.legend()

ax2.set_title('Heart Rate')



ax3.plot(power_df['cum_time'], power_df['cadence'], '.')

ax3.plot(power_df['cum_time'], power_df['cadence'].rolling(100, center=True).mean(), '-', label='smoothed')

ax3.legend()

ax3.set_title('Cadence')
sns.regplot(x='power', y='hr', data=power_df, x_bins=20, lowess=True, truncate=False)
sns.regplot(x='cadence', y='power', data=power_df, x_bins=100, lowess=True, label=True, truncate=False)
power_df.groupby(pd.qcut(power_df['cadence'], 20, duplicates='drop')).agg({'cadence': 'mean', 'power': 'mean'}).plot('cadence', 'power')
power_df.groupby(pd.qcut(power_df['cadence'], 20, duplicates='drop')).agg({'cadence': 'mean', 'hr': 'mean'}).plot('cadence', 'hr')
power_df.groupby(pd.qcut(power_df['cadence'], 20, duplicates='drop')).agg({'cadence': 'mean', 'combined_pedal_smoothness': 'mean'}).plot('cadence', 'combined_pedal_smoothness')
from scipy.interpolate import interp1d

power_vec = power_df[['cum_time', 'power', 'hr']].query('power>0').dropna().values

lin_time = np.arange(power_vec[:, 0].min(), power_vec[:, 0].max(), 1)

power_curve = interp1d(power_vec[:, 0], power_vec[:, 1])(lin_time)

hr_curve = interp1d(power_vec[:, 0], power_vec[:, 2])(lin_time)
power_vec[:, 0].min(), power_vec[:, 0].max()
plt.plot(power_curve, hr_curve, '.')

plt.xlabel('Power')

plt.ylabel('HR')
corr_df = pd.DataFrame([{'t': i, 

               'corr': np.corrcoef(power_curve[:-i], 

                                   hr_curve[i:])[0, 1]}

 for i in range(1, 50)])

corr_df.plot('t', 'corr')    
t_offset = corr_df['t'].values[np.argmax(corr_df['corr'].values)]

print('HR lags power by {} seconds'.format(t_offset))

plt.plot(power_curve[:-t_offset],

         hr_curve[t_offset:], 

         '.'

        )

plt.xlabel('Power')

plt.ylabel('HR')