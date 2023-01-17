import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np
def model_1(input, task_id=''):

    input = np.asarray(input)

    if input.shape != (2,2):

        return input.tolist()

    output = np.zeros((6,6))

    output[0,:] = np.array(input[0,:].tolist()*3)

    output[1,:] = np.array(input[1,:].tolist()*3)

    output[2,:] = np.array(input[0,:].tolist()[::-1]*3)

    output[3,:] = np.array(input[1,:].tolist()[::-1]*3)

    output[4,:] = np.array(input[0,:].tolist()*3)

    output[5,:] = np.array(input[1,:].tolist()*3)

    print('model1', task_id)

    return output.astype(int).tolist()



def model_2(task, task_id=''):

    train_data = task['train']

    test_data = task['test']

    for data in train_data:

        input = data['input']

        output = data['output']

        if np.asarray(input).shape !=np.asarray(output).shape :

            return input

    color_map = dict()

    for item in train_data:

        input = item['input']   

        output = item['output']

        for x,y in zip(input[0],output[0]):

            color_map[x] = y

    for data in test_data:

        input = data['input']

        output = np.array(input, dtype='int')

#         print('start', output)

        for i in range(output.shape[0]):

            for j in range(output.shape[1]):

                output[i,j] = color_map.get(output[i,j], 0)

    print(task_id, 'model2 predict')

    return output.tolist()

        

    


def load_task(filename, task_type=0):

    path = training_path

    if task_type==1:

        path = evaluation_path

    if task_type==2:

        path = test_path

    task_file = str(path / filename)

    with open(task_file, 'r') as f:

        task = json.load(f)

        return task

    



# ref: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines

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
from pathlib import Path



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))

print(training_tasks[:3])
ret = []

for file in training_tasks:

    task = load_task(file, 0)

    x = task['train'][0]['input']

    x = np.asarray(x)

    ret.append((file, str(x.shape)))

    

df_shape = pd.DataFrame(ret, columns=['filename','shape'])
print(df_shape.shape)
df_shape.head()
df_shape['shape'].value_counts()
file1 = df_shape[df_shape['shape']=='(2, 2)']

display(file1)
file = file1.iloc[0,0]

task = load_task(file)



plot_task(task)
file = file1.iloc[1,0]

task = load_task(file)



plot_task(task)
file1 = df_shape[df_shape['shape']=='(3, 3)']

display(file1)
file = file1.iloc[0,0]

task = load_task(file)



plot_task(task)
pred = model_2(task)

print(pred)

print(task['test'][0]['output'])
file = file1.iloc[1,0]

task = load_task(file)

print(task)

plot_task(task)
display(task['train'][0]['input'])

display(task['train'][0]['output'])
submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')

display(submission.head())
def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

display(example_grid)

print(flattener(example_grid))
for output_id in submission.index:

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:

        task = json.load(read_file)

    # skipping over the training examples, since this will be naive predictions

    # we will use the test input grid as the base, and make some modifications

    data = task['test'][pair_id]['input'] # test pair input

    # for the first guess, predict that output is unchanged

    pred_1 = model_1(data, task_id)

    pred_1 = flattener(pred_1)

    # for the second guess, change all 0s to 5s

    data = model_2(task, task_id)

    pred_2 = flattener(data)

    # for the last gues, change everything to 0

    data = [[0 for i in j] for j in data]

    pred_3 = flattener(data)

    # concatenate and add to the submission output

    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 

    submission.loc[output_id, 'output'] = pred



submission.head()
submission.to_csv('submission.csv')