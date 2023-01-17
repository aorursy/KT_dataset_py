import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

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

df_train = pd.read_csv("../input/data-without-drift/train_clean.csv")
df_test  = pd.read_csv("../input/data-without-drift/test_clean.csv")
fig, axes = create_axes_grid(1,1,20,10)
set_axes(axes, x_val=[0,len(df_train),500000,100000], y_val=[-6,12,1,1])
axes.set_title('Initial Train Signal')
axes.plot(df_train['open_channels'], color='red', linewidth=0.8);
axes.plot(df_train['signal'], color='darkblue', linewidth=0.2);

offset = 2.74
df_train['batch'] = df_train.index // 500000
df_train['modified_signal'] = df_train['signal'] + offset

mean_by_channel_per_batch = df_train.groupby(['batch', 'open_channels'])['modified_signal'].mean()
df_train['channel_means'] = df_train[['batch', 'open_channels']].apply(lambda x: mean_by_channel_per_batch[x[0],x[1]], axis=1)
fig, axes = create_axes_grid(1,1,20,10)
set_axes(axes, x_val=[0,len(df_train),500000,100000], y_val=[-6,12,1,1])
axes.set_title('Mean Signal per Open Channel per Batch')
axes.scatter(np.arange(len(df_train)), df_train['channel_means'], color='darkblue', linewidth=0.2)
#Batches 5 and 10
df_train['modified_signal'] = df_train[['batch', 'modified_signal']].apply(lambda x: (x[1]+offset) if ((x[0] == 4) or (x[0] == 9)) else x[1], axis=1)
#Batch 8
df_train['modified_signal'] = df_train[['batch', 'modified_signal']].apply(lambda x: (1.7+((x[1]-1.7)*0.93)) if (x[0] == 7) else x[1], axis=1)

visual_factor = 0.81
df_train['modified_signal'] = visual_factor * df_train['modified_signal']

print('f1_score', round(f1_score(df_train['open_channels'], np.clip(np.round(df_train['modified_signal']), 0, 10), average='macro'),3))
mean_by_channel_per_batch = df_train.groupby(['batch', 'open_channels'])['modified_signal'].mean()
df_train['channel_means'] = df_train[['batch', 'open_channels']].apply(lambda x: mean_by_channel_per_batch[x[0],x[1]], axis=1)

fig, axes = create_axes_grid(1,1,20,10)
set_axes(axes, x_val=[0,len(df_train),500000,100000], y_val=[-6,12,1,1])
axes.set_title('Updated Mean Signal per Open Channel per Batch')
axes.scatter(np.arange(len(df_train)), df_train['channel_means'], color='darkblue', linewidth=0.2)
