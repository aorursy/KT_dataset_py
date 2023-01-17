import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

train_time   = df_train["time"].values.reshape(-1,500000)
train_signal = df_train["signal"].values.reshape(-1,500000)
train_opench = df_train["open_channels"].values.reshape(-1,500000)
def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
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
fig, axes = create_axes_grid(1,2,20,5)
set_axes(axes[0], x_val=[0,500,50,10], y_val=[-5,15,5,1])
set_axes(axes[1], x_val=[0,500,50,10], y_val=[-1,11,1,1])

axes[0].set_title('training data')
axes[0].set_xticklabels('')
axes[1].set_xlabel('time / s')
axes[0].set_ylabel('signal')
axes[1].set_ylabel('open channels')

for i in range(10):
    if i in [0,2,3,4,5]:
        col = 'red'
    else:
        col = 'blue'
    axes[0].plot(train_time[i], train_signal[i], color=col, linewidth=0.1);
    axes[1].plot(train_time[i], train_opench[i], color=col, linewidth=0.1);
def markov_p(data):
    channel_range = np.unique(data)
    channel_bins = np.append(channel_range, 11)
    data_next = np.roll(data, -1)
    matrix = []
    for i in channel_range:
        current_row = np.histogram(data_next[data == i], bins=channel_bins)[0]
        current_row = current_row / np.sum(current_row)
        matrix.append(current_row)
    return np.array(matrix)
p03 = markov_p(train_opench[3])

fig, axes = create_axes_grid(1,1,10,5)
axes.set_title('Markov Transition Matrix P for sequence 3')
sns.heatmap(
    p03,
    annot=True, fmt='.3f', cmap='Blues', cbar=False,
    ax=axes, vmin=0, vmax=0.5, linewidths=2);
eig_values, eig_vectors = np.linalg.eig(np.transpose(p03))
print("Eigenvalues :", eig_values)
dist03 = eig_vectors[:,0] / np.sum(eig_vectors[:,0])
print("Probability distribution for sequence 3 :", dist03)
np.histogram(train_opench[3], bins=[0,1,2,3,4], density=True)[0]
data = train_signal[3]

fig, axes = create_axes_grid(1,1,10,10)
set_axes(axes, x_val=[-4,2,1,.1], y_val=[-4,2,1,.1])

axes.set_aspect('equal')
axes.scatter(np.roll(data,-1), data, s=.01);
data = train_signal[4]
data_true = train_opench[4]

fig, axes = create_axes_grid(1,1,10,10)
set_axes(axes, x_val=[-4,8,1,.1], y_val=[-4,8,1,.1])

axes.set_aspect('equal')
for i in range(11):
    axes.scatter(np.roll(data,-1)[data_true==i], data[data_true==i], s=.01);
