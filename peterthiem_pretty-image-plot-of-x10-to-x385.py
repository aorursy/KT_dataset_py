import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing



train = pd.read_csv("../input/train.csv")



# Add column to rank by medians

map_X0_y_rank = dict(train.groupby('X0')['y'].mean().rank())

train['X0_y_rank'] = train['X0'].apply(lambda x: map_X0_y_rank[x])
# Sort data

plot_data = train.copy()

plot_data = plot_data.sort_values(['X0_y_rank', 'y', 'ID'])

plot_data.reset_index(drop=True)



# Add alternating background shading for each X0 band

plot_data['Z_bg'] = plot_data['X0_y_rank'].apply(lambda x: 0 if x % 2 == 0 else 0.3)

for c in plot_data.loc[:,'X10':'X385'].columns:

    plot_data[c] = pd.concat([plot_data[c], plot_data['Z_bg']], axis=1).max(axis=1)                                           



# Select the columns to plot

plot_data = plot_data.loc[:,'X10':'X385']



# Plot it

plt.gray()

plt.figure(figsize=(12,300))

plt.imshow(plot_data)



plot_data = train.copy()

plot_data = plot_data.sort_values(['X0_y_rank', 'ID'])

plot_data.reset_index(drop=True)



plot_data['Z_bg'] = plot_data['X0_y_rank'].apply(lambda x: 0 if x % 2 == 0 else 0.3)

                                           

for c in plot_data.loc[:,'X10':'X385'].columns:

    plot_data[c] = pd.concat([plot_data[c], plot_data['Z_bg']], axis=1).max(axis=1)                                           



plot_data = plot_data.loc[:,'X10':'X385']



plt.gray()

plt.figure(figsize=(12,300))

plt.imshow(plot_data)


