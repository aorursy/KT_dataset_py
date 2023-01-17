

import numpy as np 

import pandas as pd

from scipy import stats

from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import seaborn as sns

from pathlib import Path

from tqdm import tqdm_notebook, trange, tqdm

from PIL import Image, ImageDraw

import io
def load_multiple_csv(input_dir, sampling=0.001):

    input_dir = Path(input_dir)

    dfs = []

    for input_path in tqdm_notebook(input_dir.glob('*.csv')):

        df = pd.read_csv(input_path)

        df = df.sample(int(len(df)*sampling))

        dfs.append(df)

    

    df = pd.concat(dfs)

    del dfs

    return df

        
# agg_df = load_multiple_csv('../input/pubg-match-deaths/aggregate',)

death_df = load_multiple_csv('../input/pubg-match-deaths/deaths')
death_df
erangel_img = Image.open('../input/pubg-match-deaths/erangel.jpg')
erangel_img
zero_coor_df = death_df[((death_df['victim_position_x'] == 0) & (death_df['victim_position_y'] == 0)) 

                        | ((death_df['killer_position_x'] == 0) & (death_df['killer_position_y'] == 0))]

zero_coor_df
def filter_outlier_death(death_df):

    

    death_df = death_df[(death_df['victim_position_x'] != 0) | (death_df['victim_position_y'] != 0)]

    death_df = death_df[(death_df['killer_position_x'] != 0) | (death_df['killer_position_y'] != 0)]

    return death_df

death_df = filter_outlier_death(death_df)


def adjust_coordinate_df(death_df):

    coordinate_columns = ['victim_position_x', 'victim_position_y',

                          'killer_position_x', 'killer_position_y']

    death_df[coordinate_columns] = death_df[coordinate_columns] / 800000 * 4096

    return death_df





death_df = adjust_coordinate_df(death_df)
reds_color = np.zeros((100, 4))

reds_color[:, 0] = np.linspace(0, 1, 100)

reds_color[:, 3] = np.linspace(0.4, 1, 100)

reds_color[:10] = 0

alpha_reds_cmap = mcolors.LinearSegmentedColormap.from_list('alpha_reds', colors=reds_color, N=1024, gamma=1)



greens_color = np.zeros((100, 4))

greens_color[:, 1] = np.linspace(0, 1, 100)

greens_color[:, 3] = np.linspace(0.4, 1, 100)

greens_color[:10] = 0

alpha_greens_cmap = mcolors.LinearSegmentedColormap.from_list('alpha_greens', colors=greens_color, N=1024, gamma=1)

def generate_heatmap(df,

                     img,

                     n_grid=100,

                     bw=200,

                     kernel='epanechnikov',

                     cmap=plt.cm.Reds,

                     alpha=None,

                     ax=None,

                    ):

    x_mesh = np.linspace(0, 4096, n_grid)

    y_mesh = np.linspace(0, 4096, n_grid)

    X, Y = np.meshgrid(x_mesh, y_mesh)

    xy = np.vstack([X.ravel(), Y.ravel()]).T



    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(df.to_numpy())

    log_density = kde.score_samples(xy)

    log_density = log_density.reshape((n_grid, n_grid))

    density = np.exp(log_density)



    if ax is None:

        fig, ax = plt.subplots(figsize=(16, 16))



    plt.gca().invert_yaxis()

    ax.contourf(X, Y, density, alpha=alpha, cmap=cmap)

    ax.imshow(img)

    return fig, ax
erangel_death_df = death_df[death_df['map'] == 'ERANGEL']

position_df = erangel_death_df[['victim_position_x', 'victim_position_y']]


fig, ax = generate_heatmap(position_df, erangel_img, bw=200, cmap=alpha_reds_cmap)
erangel_death_df = death_df[death_df['map'] == 'ERANGEL']

def euclidian_distance(a, b):

    return np.sqrt(((a - b)**2).sum(axis=1))

erangel_death_df['distance'] = euclidian_distance(erangel_death_df[['killer_position_x', 'killer_position_y']].to_numpy(),

                                                 erangel_death_df[['victim_position_x', 'victim_position_y']].to_numpy())
longrange_death_df = erangel_death_df[erangel_death_df['distance'] > erangel_death_df['distance'].quantile(0.8)]
ax = generate_heatmap(longrange_death_df[['killer_position_x', 'killer_position_y']],

                     erangel_img,

                    bw=80,

                    cmap=alpha_greens_cmap,

                     )
generate_heatmap(longrange_death_df[['victim_position_x', 'victim_position_y']],

                 erangel_img,

                bw=80,

                cmap=alpha_reds_cmap,

                )
erangel_death_df = death_df[death_df['map'] == 'ERANGEL']

worst_victim_df = erangel_death_df[erangel_death_df['victim_placement'] > 50][['victim_position_x', 'victim_position_y']]

generate_heatmap(worst_victim_df, erangel_img, bw=100, cmap=alpha_reds_cmap)
erangel_death_df = death_df[death_df['map'] == 'ERANGEL']

top_killer_df = erangel_death_df[erangel_death_df['killer_placement'] < 5][['killer_position_x', 'killer_position_y']]

generate_heatmap(top_killer_df, erangel_img, bw=100, cmap=alpha_reds_cmap)