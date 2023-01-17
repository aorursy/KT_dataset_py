import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    

from pathlib import Path



import copy



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



skill_df = pd.read_csv('/kaggle/input/task-tagging/training_tasks_tagged.csv')

skill_df.drop(['Unnamed: 0'], axis=1, inplace=True)

skill_df['task_name'] = skill_df['task_name'].apply(lambda x: x.strip('.json'))

display(skill_df.head())



# Credit to @boliu0 for these

# https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines

def plot_one(task, ax, i,train_or_test,input_or_output):

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
example_str = skill_df.iloc[82]['task']

exec("example = " + example_str)

plot_task(example)
def upside_down(task):

    new_task = copy.deepcopy(task)

    for i in range(len(task['train'])):

        pair = task['train'][i]

        new_pair = pair.copy()

        new_pair['input'] = pair['input'][::-1]

        new_pair['output'] = pair['output'][::-1]

        new_task['train'][i] = new_pair

    for i in range(len(task['test'])):

        pair = task['test'][i]

        new_pair = pair.copy()

        new_pair['input'] = pair['input'][::-1]

        new_pair['output'] = pair['output'][::-1]

        new_task['test'][i] = new_pair

    

    return new_task

    

plot_task(upside_down(example))
def horizontal_reflection(task):

    new_task = copy.deepcopy(task)

    for i in range(len(task['train'])):

        pair = task['train'][i]

        new_pair = pair.copy()

        new_pair['input'] = [x[::-1] for x in pair['input']]

        new_pair['output'] = [x[::-1] for x in pair['output']]

        new_task['train'][i] = new_pair

    for i in range(len(task['test'])):

        pair = task['test'][i]

        new_pair = pair.copy()

        new_pair['input'] = [x[::-1] for x in pair['input']]

        new_pair['output'] = [x[::-1] for x in pair['output']]

        new_task['test'][i] = new_pair

    

    return new_task

    

plot_task(horizontal_reflection(example))
def rotate_180(task):

    new_task = copy.deepcopy(task)

    new_task = upside_down(horizontal_reflection(task))

    return new_task



plot_task(rotate_180(example))
def diagonal_reflection(task):

    new_task = copy.deepcopy(task)

    for i in range(len(task['train'])):

        pair = task['train'][i]

        new_pair = pair.copy()

        new_pair['input'] = [list(x) for x in list(np.transpose(pair['input']))]

        new_pair['output'] = [list(x) for x in list(np.transpose(pair['output']))]

        new_task['train'][i] = new_pair

    for i in range(len(task['test'])):

        pair = task['test'][i]

        new_pair = pair.copy()

        new_pair['input'] = [list(x) for x in list(np.transpose(pair['input']))]

        new_pair['output'] = [list(x) for x in list(np.transpose(pair['output']))]

        new_task['test'][i] = new_pair

    

    return new_task

    

plot_task(diagonal_reflection(example))
def other_diagonal_reflection(task):

    new_task = copy.deepcopy(task)

    new_task = rotate_180(diagonal_reflection(task))

    

    return new_task



plot_task(other_diagonal_reflection(example))
def rotate_90(task):

    new_task = copy.deepcopy(task)

    new_task = upside_down(diagonal_reflection(task))

    

    return new_task



plot_task(rotate_90(example))
def rotate_270(task):

    new_task = copy.deepcopy(task)

    new_task = horizontal_reflection(diagonal_reflection(task))

    

    return new_task



plot_task(rotate_270(example))
def color_permutation(task, sigma):

    """

    Permute colors according to the rule

    x |-> sigma[x]

    """

    new_task = copy.deepcopy(task)

    for i in range(len(task['train'])):

        pair = task['train'][i]

        new_pair = pair.copy()

        new_pair['input'] = [[sigma[x] for x in row] for row in pair['input']]

        new_pair['output'] = [[sigma[x] for x in row] for row in pair['output']]

        new_task['train'][i] = new_pair

    for i in range(len(task['test'])):

        pair = task['test'][i]

        new_pair = pair.copy()

        new_pair['input'] = [[sigma[x] for x in row] for row in pair['input']]

        new_pair['output'] = [[sigma[x] for x in row] for row in pair['output']]

        new_task['test'][i] = new_pair

    

    return new_task



plot_task(color_permutation(example, [0,9,8,5,6,7,4,3,2,1]))

plot_task(color_permutation(example, [0,4,7,6,5,9,3,2,1,8]))

plot_task(color_permutation(example, [0,5,3,4,2,6,1,8,9,7]))
task_names_list = sorted(os.listdir(training_path))

task_list = []

for task_name in task_names_list: 

    task_file = str(training_path / task_name)

    with open(task_file, 'r') as f:

        task = json.load(f)

        task_list.append(task)



tasks_rotated_90 = [rotate_90(task) for task in task_list]

tasks_rotated_180 = [rotate_180(task) for task in task_list]

tasks_rotated_270 = [rotate_270(task) for task in task_list]



tasks_upside_down = [upside_down(task) for task in task_list]

tasks_reflected_hor = [horizontal_reflection(task) for task in task_list]

tasks_reflected_diag_1 = [diagonal_reflection(task) for task in task_list]

tasks_reflected_diag_2 = [other_diagonal_reflection(task) for task in task_list]



tasks_color_perm_1 = [color_permutation(task, [0,9,8,5,6,7,4,3,2,1]) for task in task_list]

tasks_color_perm_2 = [color_permutation(task, [0,4,7,6,5,9,3,2,1,8]) for task in task_list]

tasks_color_perm_3 = [color_permutation(task, [0,5,3,4,2,6,1,8,9,7]) for task in task_list]
list_of_lists = [

    (tasks_rotated_90, '_rotated_90'),

    (tasks_rotated_180, '_rotated_180'),

    (tasks_rotated_270, '_rotated_270'),

    (tasks_upside_down, '_upside_down'),

    (tasks_reflected_hor, '_reflected_hor'),

    (tasks_reflected_diag_1, '_reflected_diag_1'),

    (tasks_reflected_diag_2, '_reflected_diag_2'),

    (tasks_color_perm_1, '_color_perm_1'),

    (tasks_color_perm_2, '_color_perm_2'),

    (tasks_color_perm_3, '_color_perm_3')

]



new_tasks_df = pd.DataFrame()

for pair in list_of_lists:

    L = pair[0]

    name = pair[1]

    df = skill_df.copy()

    df['task'] = L

    df['task_name'] = df['task_name'].apply(lambda x: x + name)

    new_tasks_df = pd.concat([new_tasks_df, df], ignore_index=True)



new_tasks_df.to_csv('new_tasks_with_tags.csv')    

    

new_tasks_df
for pair in list_of_lists:

    L = pair[0]

    plot_task(np.random.choice(L))