import numpy as np

import pandas as pd

import sys

import os

import json



import matplotlib

import matplotlib.pyplot as plt

from matplotlib import colors, cm

import numpy as np
PATH_DATA = '/kaggle/input/abstraction-and-reasoning-challenge/'

PATH_training = os.path.join(PATH_DATA, 'training')

PATH_evaluation = os.path.join(PATH_DATA, 'evaluation')

PATH_test = os.path.join(PATH_DATA, 'test')



training_tasks = sorted(os.listdir(PATH_training))

evaluation_tasks = sorted(os.listdir(PATH_evaluation))
def in_out_change(image_matrix_in,

                 image_matrix_out):

    """ Calculate the difference between input and output image.

    (Can have different formats)

    """

    x_in, y_in = image_matrix_in.shape

    x_out, y_out = image_matrix_out.shape

    min_x = min(x_in, x_out)

    min_y = min(y_in, y_out)

    image_matrix_diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))

    image_matrix_diff[:x_in, :y_in] -= image_matrix_in

    image_matrix_diff[:x_out, :y_out] += image_matrix_out

    return image_matrix_diff





def plot_one(ax, image_matrix, title):

    """ Plot single example from list or array.

    """

    

    # Define colormap from -9m to +9 (negative values with lower alpha)

    cmap = colors.ListedColormap(

        ['#870C25', '#7FDBFF', '#FF851B', '#F012BE', '#AAAAAA', 

         '#FFDC00', '#2ECC40', '#FF4136', '#0074D9', 

         '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    

    cmap = np.array([colors.to_rgba(x) for x in cmap.colors]) #hex to rgba

    cmap[:,-1][:9] = 0.3 #change alpha

    cmap = colors.LinearSegmentedColormap.from_list('my_colormap', cmap)

    norm = colors.Normalize(vmin=-9, vmax=9)

    

    ax.imshow(image_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(image_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(image_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(title) 

    

    

def plot_task(task, show_difference = True):

    """

    Plots the both training and test pairs of a specified task.

    

    Args:

    --------

    show_difference: bool

        If true, plot difference between input and output image.

    """    

    num_train = len(task['train'])

    if show_difference:

        plots_per_instance = 3

    else:

        plots_per_instance = 2

    fig, axs = plt.subplots(plots_per_instance, num_train, figsize=(3*num_train,3*plots_per_instance))

    

    for i in range(num_train):

        image_matrix_in = np.array(task['train'][i]['input'])

        plot_one(axs[0,i],image_matrix_in, title = 'train - input')

        

        image_matrix_out = np.array(task['train'][i]['output'])

        plot_one(axs[1,i],image_matrix_out, title = 'train - output')

        

        if show_difference:

            image_matrix_diff = in_out_change(image_matrix_in,

                                              image_matrix_out)

            plot_one(axs[2,i],image_matrix_diff, title = 'train - difference')       

        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        image_matrix = task['test'][0]['input']

        plot_one(axs[0],image_matrix, title = 'test - input')

        image_matrix = task['test'][0]['output']

        plot_one(axs[1],image_matrix, title = 'test - output')     

    else:

        for i in range(num_test):      

            image_matrix = task['test'][i]['input']

            plot_one(axs[0,i],image_matrix, title = 'test - input')

            image_matrix = task['test'][i]['output']

            plot_one(axs[1,i],image_matrix, title = 'test - output')  

    plt.tight_layout()

    plt.show() 
#for i in range(len(training_tasks)):

for i in [1,3,4,7,8, 14, 19, 24, 26, 32, 40, 50, 79, 80, 172, 192, 278, 366]:

    task_file = os.path.join(PATH_training, training_tasks[i])

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    print(i, "ARC data sample:", training_tasks[i], 20*"--")

    plot_task(task)