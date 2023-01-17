# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    

from pathlib import Path
from pathlib import Path

path = Path('/kaggle/input/abstraction-and-reasoning-challenge')

train_df = path/'training'

evaluation_df = path/'evaluation'

test_df = path/'test'

train_task = sorted(os.listdir(train_df))

test_task = sorted(os.listdir(test_df))

evaluation_task = sorted(os.listdir(evaluation_df))
print(train_task[:4])
print(len(train_task), len(test_task), len(evaluation_task))
def plot_one(ax, i,train_or_test,input_or_output):

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

        plot_one(axs[0,i],i,'train','input')

        plot_one(axs[1,i],i,'train','output')        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one(axs[0],0,'test','input')

        plot_one(axs[1],0,'test','output')     

    else:

        for i in range(num_test):      

            plot_one(axs[0,i],i,'test','input')

            plot_one(axs[1,i],i,'test','output')  

    plt.tight_layout()

    plt.show() 
cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

plt.figure(figsize=(4, 2), dpi=300)

plt.imshow([list(range(12))], cmap=cmap, norm=norm)

plt.xticks(list(range(12)))

plt.yticks([])

plt.show()

for i in range(len(train_task)):

    task_file = str(train_df / train_task[i])

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    plot_task(task)
for i in range(len(test_task)):

    task_file = str(test_df / test_task[i])

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    plot_task(task)
for i in range(len(evaluation_task)):

    task_file = str(evaluation_df / evaluation_task[i])

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    plot_task(task)