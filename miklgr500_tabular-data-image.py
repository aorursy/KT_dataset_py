import os



import numpy as np

import pandas as pd 

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

import numba



from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
INIT_DIR = '/kaggle/input/nfl-big-data-bowl-2020'
os.listdir(INIT_DIR)
train = pd.read_csv(os.path.join(INIT_DIR, 'train.csv'), low_memory=False)
train.head()
@numba.jit

def dist_matrix(first_vector, second_vector):

    first_length = len(first_vector)

    second_length = len(second_vector)

    

    matrix = np.ones((first_length, second_length))

    for i in range(len(first_vector)):

        for j in range(len(second_vector)):

            matrix[i, j] = np.sqrt(np.mean(np.square(first_vector[i] - second_vector[j])))

    return matrix
poss_matrixs = np.zeros((len(train.PlayId.unique()[:25]), 11, 11, 3))

a_matrixs = np.zeros((len(train.PlayId.unique()[:25]), 11, 11, 3))

s_matrixs = np.zeros((len(train.PlayId.unique()[:25]), 11, 11, 3))

dis_matrixs = np.zeros((len(train.PlayId.unique()[:25]), 11, 11, 3))



for i, unique_play_id in tqdm(enumerate(train.PlayId.unique()[:25]), total=len(poss_matrixs)):

    home_poss = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'home'),:].sort_values(by='Position')[['X', 'Y']].values

    away_poss = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'away'),:].sort_values(by='Position')[['X', 'Y']].values

    

    home_a = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'home'),:].sort_values(by='Position')[['A']].values

    away_a = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'away'),:].sort_values(by='Position')[['A']].values

    

    home_s = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'home'),:].sort_values(by='Position')[['S']].values

    away_s = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'away'),:].sort_values(by='Position')[['S']].values

    

    home_dis = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'home'),:].sort_values(by='Position')[['Dis']].values

    away_dis = train.loc[(train.PlayId == unique_play_id) & (train.Team == 'away'),:].sort_values(by='Position')[['Dis']].values

    

    poss_matrixs[i] = np.dstack([dist_matrix(home_poss, home_poss), dist_matrix(away_poss, away_poss), dist_matrix(home_poss, away_poss)])

    a_matrixs[i] = np.dstack([dist_matrix(home_a, home_a), dist_matrix(away_a, away_a), dist_matrix(home_a, away_a)])

    s_matrixs[i] = np.dstack([dist_matrix(home_s, home_s), dist_matrix(away_s, away_s), dist_matrix(home_s, away_s)])

    dis_matrixs[i] = np.dstack([dist_matrix(home_dis, home_dis), dist_matrix(away_dis, away_dis), dist_matrix(home_dis, away_dis)])
poss_matrixs = (poss_matrixs - np.min(poss_matrixs, axis=0)) / (np.max(poss_matrixs, axis=0) - np.min(poss_matrixs, axis=0) + 1)

a_matrixs = (a_matrixs - np.min(a_matrixs, axis=0)) / (np.max(a_matrixs, axis=0) - np.min(a_matrixs, axis=0) + 1)

dis_matrixs = (dis_matrixs - np.min(dis_matrixs, axis=0)) / (np.max(dis_matrixs, axis=0) - np.min(dis_matrixs, axis=0) + 1)

s_matrixs = (s_matrixs - np.min(s_matrixs, axis=0)) / (np.max(s_matrixs, axis=0) - np.min(s_matrixs, axis=0) + 1)
fig = plt.figure(figsize=(25, 25))



grid = ImageGrid(fig, 111,

                 nrows_ncols=(5, 5),

                 axes_pad=0.5,

                 )



for ax, im, y in zip(grid, poss_matrixs, train.Yards.values[:22*25:22]):

    ax.imshow(im)

    ax.set_title(str(y))
fig = plt.figure(figsize=(25, 25))



grid = ImageGrid(fig, 111,

                 nrows_ncols=(5, 5),

                 axes_pad=0.5,

                 )



for ax, im, y in zip(grid, a_matrixs, train.Yards.values[:22*25:22]):

    ax.imshow(im)

    ax.set_title(str(y))
fig = plt.figure(figsize=(25, 25))



grid = ImageGrid(fig, 111,

                 nrows_ncols=(5, 5),

                 axes_pad=0.5,

                 )



for ax, im, y in zip(grid, s_matrixs, train.Yards.values[:22*25:22]):

    ax.imshow(im)

    ax.set_title(str(y))
fig = plt.figure(figsize=(25, 25))



grid = ImageGrid(fig, 111,

                 nrows_ncols=(5, 5),

                 axes_pad=0.5,

                 )



for ax, im, y in zip(grid, dis_matrixs, train.Yards.values[:22*25:22]):

    ax.imshow(im)

    ax.set_title(str(y))