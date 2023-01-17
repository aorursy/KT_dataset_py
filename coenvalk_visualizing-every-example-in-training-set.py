import os

import json

import pprint

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import colors

from matplotlib import pyplot as plt

def draw_examples(in_grids, out_grids, title):

    cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    bounds = list(range(10))

    norm = colors.BoundaryNorm(bounds, cmap.N)

    

    fig, ax = plt.subplots(nrows = len(in_grids), ncols = 2, figsize=(10, 5 * len(in_grids)))

    for idx, V in enumerate(zip(in_grids, out_grids)):

        in_grid, out_grid = V

        ax[idx][0].set_title('Example ' + str(idx) + ' Input')

        ax[idx][1].set_title('Example ' + str(idx) + ' Output')

        ax[idx][0].imshow(in_grid, cmap=cmap, norm=norm)

        ax[idx][1].imshow(out_grid, cmap=cmap, norm=norm)

    fig.suptitle(title, fontsize=20)

    plt.show()
root = '/kaggle/input/abstraction-and-reasoning-challenge'

sample_train_file = "training/017c7c7b.json"



for _, dirs, files in os.walk(os.path.join(root, "training"), topdown=False):

    for filename in files:

        with open(os.path.join(root, "training", filename), 'r') as f:

            train_sample = json.load(f)

            input_grids = []

            output_grids = []

            for example in train_sample['train']:

                input_grids.append(example['input'])

                output_grids.append(example['output'])

        

            draw_examples(input_grids, output_grids, filename)