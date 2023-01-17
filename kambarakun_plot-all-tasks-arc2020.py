%matplotlib inline
import os

import json



from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from matplotlib import colors
path_input      = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

path_training   = path_input / 'training'

path_evaluation = path_input / 'evaluation'

path_test       = path_input / 'test'
def plot_task(task):

    cmap      = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',

                                       '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm      = colors.Normalize(vmin=0, vmax=9)

    len_train = len(task['train'])

    len_test  = len(task['test'])

    len_max   = max(len_train, len_test)

    length    = {'train': len_train, 'test': len_test}

    fig, axs  = plt.subplots(len_max, 4, figsize=(15, 15*len_max//4))

    for col, mode in enumerate(['train', 'test']):

        for idx in range(length[mode]):

            axs[idx][2*col+0].axis('off')

            axs[idx][2*col+0].imshow(task[mode][idx]['input'], cmap=cmap, norm=norm)

            axs[idx][2*col+0].set_title(f"Input {mode}, {np.array(task[mode][idx]['input']).shape}")

            try:

                axs[idx][2*col+1].axis('off')

                axs[idx][2*col+1].imshow(task[mode][idx]['output'], cmap=cmap, norm=norm)

                axs[idx][2*col+1].set_title(f"Output {mode}, {np.array(task[mode][idx]['output']).shape}")

            except:

                pass

        for idx in range(length[mode], len_max):

            axs[idx][2*col+0].axis('off')

            axs[idx][2*col+1].axis('off')

    plt.tight_layout()

    plt.axis('off')

    plt.show()
for dname in [path_training, path_evaluation, path_test]:

    for fname in sorted(os.listdir(dname)):

        task_file = dname / fname

        print(task_file)

        with open(task_file, 'r') as f:

            task = json.load(f)

        plot_task(task)