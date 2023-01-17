import numpy as np

import pandas as pd

import os

import json

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from pathlib import Path
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
def show_plot(i): # t88take's funktion with improvements

    if type(i) is int: # For using slider

        task_file = str(training_path / training_tasks[i])

    else:

        task_file = str(training_path / i) # For using dropdown menu



    with open(task_file, 'r') as f:

        task = json.load(f)



    def plot_task(task):

        """

        Plots the first train and test pairs of a specified task,

        using same color scheme as the ARC app

        """

        cmap = colors.ListedColormap(

            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

        norm = colors.Normalize(vmin=0, vmax=9)

        fig, axs = plt.subplots(1, 4, figsize=(15,15))

        axs[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)

        axs[0].axis('off')

        axs[0].set_title('Train Input')

        axs[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)

        axs[1].axis('off')

        axs[1].set_title('Train Output')

        axs[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)

        axs[2].axis('off')

        axs[2].set_title('Test Input')

        axs[3].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)

        axs[3].axis('off')

        axs[3].set_title('Test Output')

        plt.tight_layout()

        plt.show()



    plot_task(task)
display(interact(show_plot, i=training_tasks))# Dropdown style
display(interact(show_plot, i=(0,400)))# Slider style