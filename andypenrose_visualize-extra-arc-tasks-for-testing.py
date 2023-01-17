import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np
# From: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines



def plot_one(task, ax, i,train_or_test, input_or_output):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    input_matrix = task[train_or_test][i][input_or_output]

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(train_or_test + ' '+input_or_output)

    



def plot_task(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one(task, axs[0,i],i,'train','input')

        plot_one(task, axs[1,i],i,'train','output')        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one(task, axs[0],0,'test','input')

        plot_one(task, axs[1],0,'test','output')     

    else:

        for i in range(num_test):      

            plot_one(task, axs[0,i],i,'test','input')

            plot_one(task, axs[1,i],i,'test','output')  

    plt.tight_layout()

    plt.show() 

extra_tasks_path = Path('/kaggle/input/extra-arc-tasks-for-testing/')



def show_extra_task(task_name):

    task_file = str(extra_tasks_path / (task_name + ".json"))    

    with open(task_file, 'r') as f:

        task = json.load(f)

    plot_task(task)
show_extra_task("new_task_01")
show_extra_task("new_task_02")
show_extra_task("new_task_03")
show_extra_task("new_task_04")
show_extra_task("new_task_05")