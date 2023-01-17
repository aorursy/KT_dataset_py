import sys

from pathlib import Path

from copy import copy

from tqdm import tqdm

import itertools



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



%matplotlib inline
N_ROWS = N_COLS = 1024

CMAP = 'Wistia'

N_SAMPLES = 500  # None - getting `MemoryError` with values larger than approx. 1_500
INPUT_DIR = Path('../input')

# print(f"Listing for directory {INPUT_DIR.name}:")

# print('\n'.join([str(f) for f in INPUT_DIR.glob("*")]))

SIIM_TRAIN_TEST = INPUT_DIR/'siim-train-test/siim'

# print(f"\nListing for directory {SIIM_TRAIN_TEST.name}:")

# print('\n'.join([str(f) for f in SIIM_TRAIN_TEST.glob("*")]))

SIIM_OTHER = INPUT_DIR/'siim-acr-pneumothorax-segmentation'

# print(f"\nListing for directory {SIIM_OTHER.name}:")

# print('\n'.join([str(f) for f in SIIM_OTHER.glob("*")]))
sys.path.insert(0, str(SIIM_OTHER))

from mask_functions import rle2mask
def diameter(mask):

    """Return the diameter of a mask.

    Note: this is a slow computation hence separated from the other statistics.

    """

    x, y = np.where(pt_mask > 0)

    pair_dist = np.sqrt((x[:, None] - x)**2 + (y[:, None] - y)**2).astype(int)

    diameter = pair_dist.max()

    return diameter



    

def mask_statistics(mask):

    """Return statistics on the geometry of a mask.  Output as a dictionary.

    """

    measure = mask.sum()



    x_dist = mask.sum(axis=0)

    x_wgts = x_dist/x_dist.sum()



    x_mean = (x_wgts*np.arange(1024)).sum()

    x_min = np.where(x_dist > 0)[0][0]

    x_max = np.where(x_dist > 0)[0][-1]



    y_dist = mask.sum(axis=1)

    y_wgts = y_dist/y_dist.sum()



    y_mean = (y_wgts*np.arange(1024)).sum()

    y_min = np.where(y_dist > 0)[0][0]

    y_max = np.where(y_dist > 0)[0][-1]



    dict = {}

    dict['measure'] = measure

    dict['mean x'] = x_mean.astype(int)

    dict['min x'] = x_min

    dict['max x'] = x_max

    dict['mean y'] = y_mean.astype(int)

    dict['min y'] = y_min

    dict['max y'] = y_max



    return dict
TRAIN_RLE = SIIM_TRAIN_TEST/'train-rle.csv'

rles = pd.read_csv(TRAIN_RLE)

rles = rles.rename(columns={' EncodedPixels': 'EncodedPixels'})

rles["EncodedPixels"] = rles["EncodedPixels"].map(lambda x: x[1:])



rles_nok = rles[rles['EncodedPixels'] != '-1'].copy()
sample = rles_nok.iloc[0]

mask = rle2mask(sample["EncodedPixels"] , N_ROWS, N_COLS).T

mask = (mask > 0).astype(int)



mask_stats = mask_statistics(mask)



print("Statistics for a sample mask:")

for key, value in mask_stats.items():

    print(f"{key:.<20}: {value:>8}")

print("\nRemark:")

print(f"{'Width of image':.<20}: {N_COLS:>8}")

print(f"{'Height of image':.<20}: {N_ROWS:>8}")

print(f"{'Measure of image':.<20}: {N_ROWS*N_COLS:>8}")
x_mean = mask_stats['mean x']

x_min = mask_stats['min x']

x_max = mask_stats['max x']

y_mean = mask_stats['mean y']

y_min = mask_stats['min y']

y_max = mask_stats['max y']

rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=3, edgecolor='r', facecolor='none')

circ = patches.Circle((x_mean, y_mean), radius=10, edgecolor='r', facecolor='k')
# Note on the use of `copy` in this cell

# A new figure is created every time the cell is run, but in matplotlib, "a single artist cannot be put in more than one figure".



fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(mask, cmap=CMAP);

ax.add_patch(copy(circ));

ax.add_patch(copy(rect));

ax.set_title("Sample mask\nwith bounding box and center of mass", fontsize=18);
all_masks = np.array(

    [rle2mask(row["EncodedPixels"] , N_ROWS, N_COLS).T

     for i, row in tqdm(itertools.islice(rles_nok.iterrows(), 0, N_SAMPLES))

    ]

)

# print(f"Calculating mean of {N_SAMPLES if N_SAMPLES is not None else len(rles_nok)} masks...")

mean_mask = all_masks.mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 8));

ax.imshow(mean_mask, cmap=CMAP);

ax.set_title(f"Superposition of {N_SAMPLES if N_SAMPLES is not None else len(rles_nok)} masks", fontsize=18);
mask_stats = pd.DataFrame()

for i, row in tqdm(itertools.islice(rles_nok.iterrows(), 0, N_SAMPLES)):

    mask = rle2mask(row["EncodedPixels"] , N_ROWS, N_COLS).T

    mask = (mask > 0).astype(int)

    mask_stats = mask_stats.append(mask_statistics(mask), ignore_index=True)

mask_stats = mask_stats.astype(int)
fig, ax = plt.subplots(figsize=(16, 6));

(mask_stats['measure'] / (N_ROWS*N_COLS)).hist(ax=ax, bins=100);

ax.set_title(f"Distribution of the measure of {N_SAMPLES if N_SAMPLES is not None else len(rles_nok)} patches", fontsize=24);

ax.set_xlabel("Percentage of pixels of entire x-ray", fontsize=14);

ax.set_ylabel("Frequency", fontsize=14);

ax.set_xticklabels([f"{item:.0%}" for item in ax.get_xticks()]);
N = 2**4

xybins = pd.DataFrame(mask_stats[['mean x', 'mean y']]).rename(columns={'mean x': 'x bin', 'mean y': 'y bin'})



xybins['x bin'] = xybins['x bin'] * N // 1024

xybins['y bin'] = xybins['y bin'] * N // 1024



xybins.insert(len(xybins.columns), "count", 1)

xybins = xybins.groupby(['x bin', 'y bin'])[['count']].count().reset_index()

xybins = xybins.pivot(columns='x bin', index='y bin', values='count').fillna(0).astype(int)
bins = N



grid_sz = 6



fig = plt.figure(figsize=(12, 12))



ax_2d = plt.subplot2grid((grid_sz, grid_sz), (0, 1), rowspan=grid_sz-1, colspan=grid_sz-1);

sns.heatmap(xybins, cmap='Blues', cbar=False, ax=ax_2d);

ax_2d.set_title("Distribution of center of mass", fontsize=24);

ax_2d.axis('off')



ax_y = plt.subplot2grid((grid_sz, grid_sz), (0, 0), rowspan=grid_sz-1, colspan=1);

mask_stats['mean y'].hist(orientation='horizontal', ax=ax_y, bins=bins);

ax_y.set_ylabel("Distribution of y-coordinate", fontsize=14);

ax_y.invert_xaxis()

ax_y.invert_yaxis()

ax_y.set_ylim([N_ROWS, 0]);

ax_y.set_yticks(range(0, N_ROWS+1, 2**6));



ax_x = plt.subplot2grid((grid_sz, grid_sz), (grid_sz-1, 1), rowspan=1, colspan=grid_sz-1);

mask_stats['mean x'].hist(ax=ax_x, bins=bins);

ax_x.set_xlabel("Distribution of x-coordinate", fontsize=14);

ax_x.invert_yaxis()

ax_x.set_xlim([0, N_COLS]);

ax_x.set_xticks(range(0, N_COLS+1, 2**6));




