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
import os

import json

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib import colors
data_dir = '/kaggle/input/abstraction-and-reasoning-challenge/'

train_dir = data_dir + 'training/'

evaluation_dir = data_dir + 'evaluation/'

test_dir = data_dir + 'test/'
training_tasks = sorted(os.listdir(train_dir))

evaluation_tasks = sorted(os.listdir(evaluation_dir))

test_tasks = sorted(os.listdir(test_dir))
train = []

for i in range(len(training_tasks)):

    task_file = train_dir + training_tasks[i]

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        train.append(task)
evaluation = []

for i in range(len(evaluation_tasks)):

    task_file = evaluation_dir + evaluation_tasks[i]

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        evaluation.append(task)
test = []

for i in range(len(test_tasks)):

    task_file = test_dir + test_tasks[i]

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        test.append(task)
print('Our dataset has {} training examples.'.format(len(train)))

print('Our dataset has {} evaluation examples.'.format(len(evaluation)))

print('Our dataset has {} test examples.'.format(len(test)))
train_count = 0

for i in range(len(train)):

    train_count += len(train[i]['train'])

print('Our training dataset has {} training examples.'.format(train_count))
def plot_task(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    fig, axs = plt.subplots(1, 6, figsize=(10,10))

    axs[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)

    axs[0].axis('off')

    axs[0].set_title('Train Input')

    axs[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)

    axs[1].axis('off')

    axs[1].set_title('Train Output')

    axs[5].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)

    axs[5].axis('off')

    axs[5].set_title('Test Input')

    axs[4].imshow(task['test'][0]['output'], cmap=cmap, norm=norm)

    axs[4].axis('off')

    axs[4].set_title('Test Output')

    axs[2].imshow(task['train'][1]['input'], cmap=cmap, norm=norm)

    axs[2].axis('off')

    axs[2].set_title('Train Input')

    axs[3].imshow(task['train'][1]['output'], cmap=cmap, norm=norm)

    axs[3].axis('off')

    axs[3].set_title('Train Output')

    plt.tight_layout()

    plt.show()
plot_task(train[10])