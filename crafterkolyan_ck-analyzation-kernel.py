import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

import os
def process_date(x, start_time=[]):
    x = x.copy()
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in x['date']]
    SECONDS_IN_DAY = 24 * 60 * 60
    if not start_time:
        start_time.append(min(dates).timestamp() / SECONDS_IN_DAY)
    start_time = start_time[0]
    x['timestamp'] = [x.timestamp() / SECONDS_IN_DAY - start_time for x in dates]
    x['year'] = [x.year for x in dates]
    x['month'] = [x.month for x in dates]
    x['day'] = [x.day for x in dates]
    x['weekday'] = [x.weekday() for x in dates]
    x.drop('date', axis=1, inplace=True)
    return x
def plots(x, y, figsize=(15, 5), cols=6, plot_function=matplotlib.axes.Axes.scatter):
    x_names = x.columns if len(x.shape) > 1 else np.array([x.name])
    y_names = y.columns if len(y.shape) > 1 else np.array([y.name])
    rows = (x_names.size * y_names.size + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(x_names.size * y_names.size):
        if rows == 1:
            if cols == 1:
                cur_ax = ax
            else:
                cur_ax = ax[i % cols]
        else:
            cur_ax = ax[i // cols][i % cols]
        x_name = x_names[i % x_names.size]
        y_name = y_names[i // x_names.size]
        cur_x = x[x_name] if x_names.size > 1 else x
        cur_y = y[y_name] if y_names.size > 1 else y
        cur_ax.set_xlabel(x_name, fontsize=10)
        cur_ax.set_ylabel(y_name, fontsize=10)
        plot_function(cur_ax, cur_x, cur_y, c='b')
    
    fig.tight_layout()
    plt.show()
x_train = pd.read_csv('../input/train_data.csv', index_col='index')
y_train = pd.read_csv('../input/train_target.csv', index_col='index')
x_test = pd.read_csv('../input/test_data.csv', index_col='index')
x_train = process_date(x_train)
x_test = process_date(x_test)
plots(x_train, y_train, figsize=(25, 10))
X = x_train.copy()
X['price'] = y_train
X['price'].describe()
gby = X.groupby('timestamp')
price_stat = gby['price'].agg(['mean', 'min', 'max', 'count'])
plots(price_stat.index, price_stat, figsize=(25, 5), cols=price_stat.shape[1], plot_function=matplotlib.axes.Axes.plot)
gby = X.groupby('month')
price_stat = gby['price'].agg(['mean', 'min', 'max', 'count'])
plots(price_stat.index, price_stat, figsize=(25, 5), cols=price_stat.shape[1], plot_function=matplotlib.axes.Axes.plot)
gby = X.groupby('weekday')
price_stat = gby['price'].agg(['mean', 'min', 'max', 'count'])
plots(price_stat.index, price_stat, figsize=(25, 5), cols=price_stat.shape[1], plot_function=matplotlib.axes.Axes.plot)
corr = X.corr()
corr
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})