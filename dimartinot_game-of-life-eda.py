import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from tqdm.notebook import tqdm
train_csv = pd.read_csv("../input/conways-reverse-game-of-life-2020/train.csv")
train_csv.head()
col = train_csv.columns

start_col_filtered = [c for c in col if "start_" in c]

end_col_filtered = [c for c in col if "stop_" in c]

start_grids = train_csv[start_col_filtered].to_numpy()

stop_grids = train_csv[end_col_filtered].to_numpy()
def get_grid(list_of_cells, shape=(25,25), plot=True):

    mat = list_of_cells.reshape(shape)

    if (plot):

        sns.heatmap(mat)

    return mat

    

def plot_grids(idx):

    start_cells, end_cells = start_grids[idx], stop_grids[idx]

    start_mat = get_grid(start_cells, plot=False)

    end_mat = get_grid(end_cells, plot=False)

    

    fig, axes = plt.subplots(1,2, figsize=(25,10))

    sns.heatmap(start_mat, ax=axes[0])

    axes[0].set_title("Starting board")

    sns.heatmap(end_mat, ax=axes[1])

    axes[1].set_title("Ending board")

    

idx = 5

plot_grids(idx)



idx = 2

plot_grids(idx)
def get_diff(idx, plot=True):

    start_cells, end_cells = start_grids[idx], stop_grids[idx]

    start_grid = get_grid(start_cells, plot=False)

    end_grid = get_grid(end_cells, plot=False)

    

    img = np.zeros((25,25,3))

    img[:,:,0] = start_grid #every red pixel was pixel present in start grid and absent in end grid

    img[:,:,1] = end_grid #every green pixel was pixel present in end grid and absent in start grid

    # every black pixel was absent in both images

    # every yellow pixel was present in both images

    diff_mat = np.zeros((25,25))

    diff_mat = start_grid + 2*end_grid

    if plot:

        plt.imshow(img)

    

    return diff_mat

    

diff_mat = get_diff(3)
values = []

for idx in tqdm(range(len(start_grids))):

    diff_mat = get_diff(idx, plot=False)

    values.extend(list(diff_mat.flatten()))
plt.hist(values)