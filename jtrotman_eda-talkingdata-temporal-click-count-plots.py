%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, time

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16'
        }

# read a subset, runs out of memory otherwise ('os' seems least interesting)
fields = [ 'ip', 'app', 'device', 'channel' ]
to_read = fields + [ 'click_time' ]

train_df  = pd.read_csv('../input/train.csv', usecols=to_read, parse_dates=['click_time'], dtype=dtypes) #, nrows=72000000)
test_df   = pd.read_csv('../input/test.csv', usecols=to_read, parse_dates=['click_time'], dtype=dtypes)
print('Loaded', train_df.shape, test_df.shape)
def datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.int32)

train_df['sec'] = datetime_to_deltas(train_df.click_time)
test_df['sec'] = datetime_to_deltas(test_df.click_time)
print('Added seconds')

train_df.drop('click_time', axis=1, inplace=True)
test_df.drop('click_time', axis=1, inplace=True)
print('Dropped click_time')
from matplotlib.colors import LogNorm

# e.g. train_df.loc[train_df.ip==234]
def generate_plot(df):
    w = 600
    n = df.sec.max()+1
    l = int(np.ceil(n/float(w))*w)
    c = np.zeros(l, dtype=np.float32)
    np.add.at(c, df.sec.values, 1)
    print(f'total clicks {c.sum():.0f} \t max clicks {c.max():.0f} \t mean click rate {c.mean():.02f} ')
    return c.reshape((-1,w))

def show(pix, title):
    pix += 1 # matplotlib restriction
    ysize = 5 if pix.shape[0] < 400 else 8
    fig, ax0 = plt.subplots(figsize=(18, ysize))
    ax0.invert_yaxis()
    ax0.set_yticks(np.arange(0, pix.shape[0], 144), False)
    ax0.set_yticks(np.arange(0, pix.shape[0], 6), True)
    ax0.set_xticks(np.arange(0, pix.shape[1], 60), False)
    ax0.set_xticks(np.arange(0, pix.shape[1], 10), True)
    c = ax0.pcolormesh(pix, norm=LogNorm(vmin=1, vmax=pix.max()), cmap='afmhot')
    ax0.set_title(title)
    return fig.colorbar(c, ax=ax0)

def gen_show(df, col, value):
    idx = df[col]==value
    if idx.sum()<1:
        print('Not found!')
    else:
        pix = generate_plot(df.loc[idx])
        show(pix, f'{col} {value}')
gen_show(train_df, 'app', 3)
gen_show(train_df, 'app', 1)
gen_show(train_df, 'app', 7)
gen_show(train_df, 'app', 20)
gen_show(train_df, 'app', 46)
gen_show(train_df, 'app', 151)
gen_show(train_df, 'app', 56)
gen_show(train_df, 'channel', 236)
gen_show(train_df, 'channel', 105)
gen_show(train_df, 'channel', 244)
gen_show(train_df, 'channel', 419)
gen_show(train_df, 'channel', 272)
gen_show(train_df, 'channel', 114)
gen_show(train_df, 'device', 2)
gen_show(train_df, 'device', 3543)
gen_show(train_df, 'device', 154)
gen_show(train_df, 'ip', 5314)
gen_show(train_df, 'ip', 5348) 
gen_show(train_df, 'ip', 86767)
gen_show(test_df, 'app', 24)
gen_show(test_df, 'app', 183)
gen_show(test_df, 'channel', 125)
gen_show(test_df, 'channel', 236)