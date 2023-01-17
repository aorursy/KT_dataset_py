import os

import time



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import lightgbm as lgb



from sklearn import model_selection, metrics



from tsfresh import extract_features
df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

df_test  = pd.read_csv("../input/liverpool-ion-switching/test.csv")
def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):

    fig, axes = plt.subplots(numplots_y, numplots_x)

    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)

    return fig, axes

    

def set_axes(axes, use_grid=True, x_val = [0,100,10,5], y_val = [-50,50,10,5]):

    axes.grid(use_grid)

    axes.tick_params(which='both', direction='inout', top=True, right=True, labelbottom=True, labelleft=True)

    axes.set_xlim(x_val[0], x_val[1])

    axes.set_ylim(y_val[0], y_val[1])

    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[2] + 1).astype(int)))

    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[3] + 1).astype(int)), minor=True)

    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[2] + 1).astype(int)))

    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[3] + 1).astype(int)), minor=True)
fig, axes = create_axes_grid(1,2,30,5)

set_axes(axes[0], x_val=[0,5000000,500000,100000], y_val=[-5,15,5,1])

axes[0].plot(df_train['signal'], color='darkblue', linewidth=.1);

axes[0].set_title('training')

set_axes(axes[1], x_val=[0,2000000,100000,10000], y_val=[-5,15,5,1])

axes[1].set_title('test')

axes[1].plot(df_test['signal'], color='darkgreen', linewidth=.1);
train_segm_separators = np.concatenate([[0,500000,600000], np.arange(1000000,5000000+1,500000)])

train_segm_signal_groups = [0,0,0,1,2,4,3,1,2,3,4] # from visual identification

train_segm_is_shifted = [False, True, False, False, False, False, False, True, True, True, True] # from visual identification

train_signal = np.split(df_train['signal'].values, train_segm_separators[1:-1])

train_opench = np.split(df_train['open_channels'].values, train_segm_separators[1:-1])
test_segm_separators = np.concatenate([np.arange(0,1000000+1,100000), [1500000,2000000]])

test_segm_signal_groups = [0,2,3,0,1,4,3,4,0,2,0,0] # from visual id

test_segm_is_shifted = [True, True, False, False, True, False, True, True, True, False, True, False] # from visual id

test_signal = np.split(df_test['signal'].values, test_segm_separators[1:-1])
#create clean signal histograms

fig, axes = create_axes_grid(1,1,30,4)

set_axes(axes, x_val=[-4,8,1,.1], y_val=[0,0.05,0.01,0.01])



clean_hist = []

hist_bins = np.linspace(-4,10,500)



# iterate through the clean signal groups 0, 3, 4, 6, 5

for j,i in enumerate([0,3,4,6,5]):

    clean_hist.append(np.histogram(train_signal[i], bins=hist_bins)[0])

    clean_hist[-1] = clean_hist[-1] / 500000 # normalize histogram

    axes.plot(hist_bins[1:], clean_hist[-1], label='Segment '+str(i)+', signal group '+str(j));

axes.legend();

axes.set_title("Clean reference histograms for all 5 signal groups");
window_size = 1000



fig, axes = create_axes_grid(1,1,30,4)

set_axes(axes, x_val=[-4,2,1,.1], y_val=[0,0.05,0.01,0.01])



axes.plot(hist_bins[1:], clean_hist[0]);

for i in [0,25000,50000,75000]:

    window_hist = np.histogram(train_signal[1][i:i+window_size], bins=hist_bins)[0] / window_size

    axes.plot(hist_bins[1:], window_hist);
window_size = 1000

bin_width = np.diff(hist_bins)[0]

s_window = 10 # maximum absolute change in shift from window to window+1

train_signal_shift = []



for clean_id in range(len(train_segm_signal_groups)):

    

    group_id = train_segm_signal_groups[clean_id]

    window_shift = []

    prev_s = 0 # all signal groups start with shift=0

    window_data = train_signal[clean_id].reshape(-1,window_size)

    

    for w in window_data:

        window_hist = np.histogram(w, bins=hist_bins)[0] / window_size

        window_corr = np.array([ np.sum(clean_hist[group_id] * np.roll(window_hist, -s)) for s in range(prev_s-s_window, prev_s+s_window+1) ])

        prev_s = prev_s + np.argmax(window_corr) - s_window

        window_shift.append(-prev_s * bin_width)



    window_shift = np.array(window_shift)

    train_signal_shift.append(window_shift)    
fig, axes = create_axes_grid(1,1,30,4)

set_axes(axes, x_val=[0,5000,500,100], y_val=[-5,1,1,.1])

axes.plot(np.concatenate(train_signal_shift));

axes.set_title("Shift value as determined by histogram matching:");
train_signal_shift_clean = []

train_signal_detrend = []



for data, use_fit, signal in zip(train_signal_shift, train_segm_is_shifted, train_signal):

    if use_fit:

        data_x = np.arange(len(data), dtype=float) * window_size + window_size/2

        fit = np.flip(np.polyfit(data_x, data, 4))

        data_x = np.arange(len(data) * window_size, dtype=float)

        data_2 = np.sum([ c * data_x ** i for i, c in enumerate(fit) ], axis=0)

    else:

        data_2 = np.zeros(len(data) * window_size, dtype=float)

        

    train_signal_shift_clean.append(data_2)

    train_signal_detrend.append(signal + data_2)
fig, axes = create_axes_grid(1,1,30,4)

set_axes(axes, x_val=[0,5000000,500000,100000], y_val=[-6,1,1,.1])

axes.plot(np.concatenate(train_signal_shift_clean));

axes.set_title("Final shift value after polynomial fit");
fig, axes = create_axes_grid(1,1,30,5)

set_axes(axes, x_val=[0,5000000,500000,100000], y_val=[-5,15,5,1])

axes.plot(np.concatenate(train_signal_detrend), linewidth=.1);

axes.set_title("Training data without shift");
window_size = 1000

bin_width = np.diff(hist_bins)[0]

s_window = 10

test_signal_detrend = []

test_signal_shift = []



for clean_id in range(len(test_segm_signal_groups)):

    

    group_id = test_segm_signal_groups[clean_id]

    window_shift = []

    prev_s = 0

    window_data = test_signal[clean_id].reshape(-1,window_size)

    

    for w in window_data:

        window_hist = np.histogram(w, bins=hist_bins)[0] / window_size

        window_corr = np.array([ np.sum(clean_hist[group_id] * np.roll(window_hist, -s)) for s in range(prev_s-s_window, prev_s+s_window+1) ])

        prev_s = prev_s + np.argmax(window_corr) - s_window

        window_shift.append(-prev_s * bin_width)



    window_shift = np.array(window_shift)

    test_signal_shift.append(window_shift)    
fig, axes = create_axes_grid(1,1,30,4)

set_axes(axes, x_val=[0,2000,100,10], y_val=[-6,1,1,.1])

axes.plot(np.concatenate(test_signal_shift));

axes.set_title("Shift value as determined by histogram matching:");
test_signal_shift_clean = []

test_signal_detrend = []

test_remove_shift = [True, True, False, False, True, False, True, True, True, False, True, False]



for data, use_fit, signal in zip(test_signal_shift, test_segm_is_shifted, test_signal):

    if use_fit:

        data_x = np.arange(len(data), dtype=float) * window_size + window_size/2

        fit = np.flip(np.polyfit(data_x, data, 4))

        data_x = np.arange(len(data) * window_size, dtype=float)

        data_2 = np.sum([ c * data_x ** i for i, c in enumerate(fit) ], axis=0)

    else:

        data_2 = np.zeros(len(data) * window_size, dtype=float)

        

    test_signal_shift_clean.append(data_2)

    test_signal_detrend.append(signal + data_2)
fig, axes = create_axes_grid(1,1,30,4)

set_axes(axes, x_val=[0,2000000,100000,10000], y_val=[-6,1,1,.1])

axes.plot(np.concatenate(test_signal_shift_clean));

axes.set_title("Final shift value after polynomial fit");
fig, axes = create_axes_grid(1,1,30,5)

set_axes(axes, x_val=[0,2000000,200000,10000], y_val=[-5,12,5,1])

axes.plot(np.concatenate(test_signal_detrend), linewidth=.1);

axes.set_title("Test data without shift");
# np.savez_compressed("train_detrend.npz", train_signal=train_signal_detrend, train_opench=train_opench, train_groups=train_segm_signal_groups)

train = df_train.copy()

train['signal_normalized'] = np.concatenate(train_signal_detrend).ravel()
# np.savez_compressed("test_detrend.npz", test_signal=test_signal_detrend, test_groups=test_segm_signal_groups)

test = df_test.copy()

test['signal_normalized'] = np.concatenate(test_signal_detrend).ravel()
rows_per_sample = 500000

train['sample'] = train.groupby(train.index // rows_per_sample).ngroup()

test['sample'] = test.groupby(test.index // rows_per_sample).ngroup()
# Feature creation taken from https://www.kaggle.com/vbmokin/ion-switching-advanced-fe-lgb-confmatrix

def create_features(df, window_sizes, group='sample', column='signal_normalized'):

    for window in window_sizes:

        df[f"rolling_mean_{window}"] = df.groupby(group)[column].rolling(window=window).mean().values

        df[f"rolling_std_{window}"] = df.groupby(group)[column].rolling(window=window).std().values

        df[f"rolling_var_{window}"] = df.groupby(group)[column].rolling(window=window).var().values

        df[f"rolling_min_{window}"] = df.groupby(group)[column].rolling(window=window).min().values

        df[f"rolling_max_{window}"] = df.groupby(group)[column].rolling(window=window).max().values

        df[f"rolling_min_max_ratio_{window}"] = df[f"rolling_min_{window}"] / df[f"rolling_max_{window}"]

        df[f"rolling_min_max_diff_{window}"] = df[f"rolling_max_{window}"] - df[f"rolling_min_{window}"]

        

        df[f'lag_diff_{window}'] = df.groupby(group)[column].diff(window)



        a = (df[column] - df[f'rolling_min_{window}']) / (df[f'rolling_max_{window}'] - df[f'rolling_min_{window}'])

        df[f"norm_{window}"] = a * (np.floor(df[f'rolling_max_{window}']) - np.ceil(df[f'rolling_min_{window}']))



    return df.replace([np.inf, -np.inf], np.nan).fillna(0)
# arbitrary assign a bunch of windows

windows = [2, 3, 5, 10, 20, 30, 50]
train = create_features(train, windows)

train = train.drop(['time', 'signal', 'sample'], axis='columns').rename(columns={'signal_normalized': 'signal'})
train.filter(regex='norm*|rolling*').iloc[10000:11000]
test = create_features(test, windows)

test = test.drop(['time', 'signal', 'sample'], axis='columns').rename(columns={'signal_normalized': 'signal'})
X, y = train.drop(['open_channels'], axis='columns'), train['open_channels']

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)
def gbdt(X_train, y_train, X_valid, y_valid, early_stopping_rounds=10, num_boost_round=150):

    # these are taken from various notebooks

    params = {

        'learning_rate': 0.1,

        'max_depth': 2 ** 7 + 1,

        'num_leaves': 200,

        'metric': 'logloss',

        'random_state': 7,

        'n_jobs': -1,

        'sample_fraction': 0.33,

    }

    

    def f1_metric(preds, train_df):

        labels = train_df.get_label()

        preds = np.round(np.clip(preds, 0, 10)).astype(int)

        score = metrics.f1_score(labels, preds, average = 'macro')

        return ('f1_metric', score, True)

    

    clf = lgb.train(params, lgb.Dataset(X_train, y_train), valid_sets=lgb.Dataset(X_valid, y_valid), early_stopping_rounds=early_stopping_rounds, num_boost_round=num_boost_round, feval=f1_metric)

    y_valid_pred = np.rint(clf.predict(X_valid,  num_iteration=clf.best_iteration))

    

    return clf, y_valid_pred
clf, y_valid_pred = gbdt(X_train, y_train, X_valid, y_valid)
metrics.f1_score(y_valid, y_valid_pred, average='macro')
lgb.plot_importance(clf, figsize=(50,50))
y_test = np.rint(clf.predict(test, num_iteration=clf.best_iteration))
submission = df_test[['time']].copy()

submission['open_channels'] = y_test.astype(int)
submission.to_csv(f"results_{time.strftime('%Y%m%d%H%M%S')}.csv", index=False, float_format='%.4f')