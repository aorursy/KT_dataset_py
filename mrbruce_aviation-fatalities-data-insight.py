%matplotlib inline
%reload_ext autoreload
%autoreload 2
from fastai.tabular import *
from pandas import *
from scipy import *
from matplotlib import gridspec
from scipy.fftpack import fft
from scipy.optimize import leastsq
from scipy import signal

import scipy.fftpack
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
df = pd.read_csv("../input/train.csv")
len(df)
df.head()
def get_data(crew, exp, seat):
    return df[(df.crew == crew) & (df.seat == seat) & (df.experiment == exp)].sort_values(by=['time'])

def trim_data(x, from_pt, secs):
    return x[(x.time >= from_pt) & (x.time < (from_pt + secs))]

crew_1_all = df[(df.crew == 1) & (df.seat == 0)].sort_values(by=['time'])
crew_1_ca = get_data(1, 'CA', 0)
crew_1_ss = get_data(1, 'SS', 0)
crew_1_da = get_data(1, 'DA', 0)
len(crew_1_all), len(crew_1_ca), len(crew_1_ss), len(crew_1_da)
(crew_1_all.time.min(), crew_1_all.time.max()), (crew_1_ca.time.min(), crew_1_ca.time.max()), (crew_1_ss.time.min(), crew_1_ss.time.max()), (crew_1_da.time.min(), crew_1_da.time.max()),
plt.plot(crew_1_ca.time, crew_1_ca.event);
plt.plot(crew_1_ss.time, crew_1_ss.event);
plt.plot(crew_1_da.time, crew_1_da.event);
crew_to_use = crew_1_ca
secs = 10
crew_trim = trim_data(crew_to_use, 120.0, secs) # 10 secs of data from 2min in
plt.plot(crew_trim.time, crew_trim.ecg);
plt.plot(crew_1_da.time, crew_1_da.ecg);
len(crew_trim), (len(crew_trim) / secs)
crew_trim.head()
plt.plot(crew_trim.time, crew_trim.ecg.rolling(8).mean());
plt.plot(crew_trim.time, crew_trim.gsr);
plt.plot(crew_to_use.time, crew_to_use.gsr);
plt.plot(crew_trim.time, crew_trim.r);
plt.plot(crew_trim.time, crew_trim.ecg.rolling(256).mean());
cleared = crew_trim.ecg.rolling(8).mean() - crew_trim.ecg.rolling(256).mean()
plt.plot(crew_trim.time, cleared);
y0 = crew_trim['eeg_f8']
y1 = crew_trim['eeg_f8'].rolling(8).mean()
x = crew_trim.time

# graph it!
fig = plt.figure()
gs = gridspec.GridSpec(2, 1)

ax0 = plt.subplot(gs[0])
line0, = ax0.plot(x, y0, color='r')

ax1 = plt.subplot(gs[1], sharex = ax0)
line1, = ax1.plot(x, y1, color='b')

plt.setp(ax0.get_xticklabels(), visible=False)

yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

plt.subplots_adjust(hspace=.0)
plt.show()
ordered_fields = [['eeg_fp1', 'eeg_f3', 'eeg_c3', 'eeg_p3', 'eeg_o1'], 
                  ['eeg_fp2', 'eeg_f4', 'eeg_c4', 'eeg_p4', 'eeg_o2'],
                  ['eeg_f7',  'eeg_t3', 'eeg_t5'],
                  ['eeg_f8',  'eeg_t4', 'eeg_t6'],
                  ['eeg_fz',  'eeg_cz', 'eeg_pz', 'eeg_poz']]

colours = ['darkslateblue', 'cadetblue', 
           'rebeccapurple', 'teal', 'chocolate']
fig = plt.figure(figsize=(12,12))
fld_cnt = sum([len(x) for x in ordered_fields])
gs = gridspec.GridSpec(fld_cnt, 1)

x = crew_trim.time
ax0 = None
axL = None

spines = ["top", "right", "left", "bottom"]

i = 0
for arr, col in zip(ordered_fields, colours):
    for fld in arr:
        ax = plt.subplot(gs[i])
        ln = ax.plot(x, crew_trim[fld].rolling(8).mean(), color=col)
        
        ax.set_yticks([])        
        ax.set(ylabel=fld.replace("eeg_", ""))
        ax.yaxis.label.set_rotation(0)
        ax.xaxis.grid(which="major")
        
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        
        for s in spines: ax.spines[s].set_visible(False)
        if i == 0: ax0 = ax
            
        axL = ax
        i = i+1

plt.setp(axL.get_xticklabels(), visible=True)
plt.subplots_adjust(hspace=.0)
plt.show()
x = crew_trim.r.values
avg = mean(x)
x = x - avg

peaks, _ = scipy.signal.find_peaks(x, distance=500)
peaks
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show();
peak_times = crew_trim.time.values[peaks]
peak_diffs = [peak_times[n] - peak_times[n-1] for n in range(1, len(peak_times))]
rolling_rr = [60/n for n in peak_diffs]

# Buff out the rolling RR as we don't have RR initially
# we just duplicate the initial value here, we could use zero if we wanted
rolling_rr.insert(0, rolling_rr[0])
peaks, peak_times, peak_diffs, rolling_rr
expanded_rr = [0] * len(x)
next_rr = 0
for i in range(0, len(x)):
    expanded_rr[i] = rolling_rr[next_rr]
    if (i == peaks[next_rr]) & (next_rr < (len(rolling_rr)-1)):
        next_rr = next_rr+1
plt.plot(crew_trim.time, expanded_rr);
def peak_rate(times, vals, d=None, p=None, w=None, h=None):
    avg = mean(vals[~np.isnan(vals)])
    vals = vals - avg
    peaks, _ = scipy.signal.find_peaks(vals, 
                                    distance=d, 
                                    prominence=p,
                                    width=w,
                                    height=h)
    peak_times = times.values[peaks]
    peak_diffs = [peak_times[n] - peak_times[n-1] for n in range(1, len(peak_times))]
    minute_rates = [60/n for n in peak_diffs]
    expanded_rates = [0] * len(vals)
    # Buff out the rolling rate as we don't have rate initially
    if len(minute_rates) > 0:
        minute_rates.insert(0, minute_rates[0])
        n = 0
        for i in range(0, len(vals)):
            expanded_rates[i] = minute_rates[n]
            if (i == peaks[n]) & (n < (len(minute_rates)-1)):
                n = n+1
    return pd.Series(expanded_rates)
def plot_rate(x, y, d=None, p=None, w=None, h=None):
    rates = peak_rate(x, y, d, p, w, h)
    # 7680 = 256*30 = average every 30 seconds?
    rolling_rate = rates.rolling(7680).mean()
    plt.plot(x, rates, color='lightgray');
    plt.plot(x, rolling_rate);
plot_rate(crew_to_use.time, crew_to_use.r, d=500)
x = crew_trim.ecg.rolling(48).mean().values #
avg = mean(x[~np.isnan(x)])
x = x - avg

peaks, _ = sp.signal.find_peaks(x, distance=64, prominence=1, width=1, height=50)
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
#plt.plot(np.zeros_like(x), "--", color="gray")
plt.show();
plot_rate(crew_to_use.time, crew_to_use.ecg, d=64, p=1, w=1)
plot_rate(crew_to_use.time, crew_to_use.ecg, d=64, p=1, w=1, h=50)

