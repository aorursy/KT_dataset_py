import numpy as np

import pandas as pd

from glob import glob

import os

import json

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib import colors
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(glob(str(training_path / '*')))

evaluation_tasks = sorted(os.listdir(evaluation_path))
def getData(task_filename):

    with open(task_filename, 'r') as f:

        task = json.load(f)

    return task
def plotOne(ax,task,i,train_or_test,input_or_output):

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

    



def plotTask(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plotOne(axs[0,i],task,i,'train','input')

        plotOne(axs[1,i],task,i,'train','output')        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plotOne(axs[0],task,0,'test','input')

        plotOne(axs[1],task,0,'test','output')     

    else:

        for i in range(num_test):      

            plotOne(axs[0,i],task,i,'test','input')

            plotOne(axs[1,i],task,i,'test','output')  

    plt.tight_layout()

    plt.show()
correct = 0

total = len(training_tasks)

unsolved = []



for task in training_tasks:

    data = getData(task)

    

    x = [np.array(x['input']) for x in data['train']]

    y = [np.array(x['output']) for x in data['train']]

    x_shapes = [x.shape for x in x]

    y_shapes = [x.shape for x in y]

    io_ratio = [(j[0] / i[0], j[1] / i[1]) for i,j in zip(x_shapes,y_shapes)]

    io_diff =  [(j[0] - i[0], j[1] - i[1]) for i,j in zip(x_shapes,y_shapes)]

    

    x_test = [np.array(x['input']) for x in data['test']]

    test_x_shapes = [x.shape for x in x_test]

    output_shapes = [(3,3)] * len(test_x_shapes)

    

    if len(list(set(io_ratio))) == 1: # Output shapes have the same input/output ratio

        io_ratio = io_ratio[0]

        output_shapes = [(shape[0] * io_ratio[0], shape[1] * io_ratio[1]) for shape in test_x_shapes]

    elif len(list(set(io_diff))) == 1: # Output shapes have the same input/output difference

        io_diff = io_diff[0]

        output_shapes = [(shape[0] + io_diff[0], shape[1] + io_diff[1]) for shape in test_x_shapes]

    elif len(list(set(y_shapes))) == 1: # Outputs have the same shape

        output_shapes = [y_shapes[0]] * len(test_x_shapes)

    

    # Check if output_shapes prediction is correct for all test examples

    y_test_shapes = [np.array(x['output']).shape for x in data['test']]

    solved = sum([1 for idx,test_shapes in enumerate(y_test_shapes) if test_shapes == output_shapes[idx]]) == len(y_test_shapes)

    if not solved: unsolved.append(task)

    correct += solved



print('%d/%d (%.1f' % (correct,total,100*correct/total) + r' %)')
for task in unsolved:

    data = getData(task)

    plotTask(data)