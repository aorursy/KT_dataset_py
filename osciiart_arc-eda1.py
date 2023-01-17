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
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))

print(len(training_tasks), len(evaluation_tasks), len(test_tasks))

cmap = colors.ListedColormap(

['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

plt.figure(figsize=(5, 2), dpi=200)

plt.imshow([list(range(10))], cmap=cmap, norm=norm)

plt.xticks(list(range(10)))

plt.yticks([])

plt.show()



def plot_task(task):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()
def get_data(task_filename):

    with open(task_filename, 'r') as f:

        task = json.load(f)

    return task



num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

color2num = {c: n for n, c in enumerate(num2color)}
def check(task, pred_func):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        t_pred = pred_func(t_in)

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)

        axs[2][fig_num].set_title(f'Train-{i} pred')

        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))

        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        t_pred = pred_func(t_in)

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)

        axs[2][fig_num].set_title(f'Test-{i} pred')

        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))

        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))

        fig_num += 1
def task_train000(x):

    x_upsampled = x.repeat(3, axis=0).repeat(3, axis=1)

    x_tiled = np.tile(x, (3, 3))

    y = x_upsampled & x_tiled

    return y



task = get_data(str(training_path / training_tasks[0]))

check(task, task_train000)
df_train = pd.DataFrame(training_tasks, columns=['id'])

df_train['id'] = df_train['id'].apply(lambda x: x[:-5])

df_train.head()
idx = 0

tmp = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

print(tmp.keys())

print(tmp['train'])

print(len(tmp['train']))

print(tmp['train'][0]['input'])

print(np.array(tmp['train'][0]['input']))

print(np.array(tmp['train'][0]['input']).shape)

print(tmp['test'])

print(tmp)
def get_num_train(x):

    data = get_data("{}/{}.json".format(training_path, x))

    ans = len(data['train'])

    return ans



df_train['num_train'] = -1

df_train['num_train'] = df_train['id'].apply(lambda x: get_num_train(x))

df_train.head()
def get_num_test(x):

    data = get_data("{}/{}.json".format(training_path, x))

    ans = len(data['test'])

    return ans



df_train['num_test'] = -1

df_train['num_test'] = df_train['id'].apply(lambda x: get_num_test(x))

df_train.head()
def get_inputshape_is_output_shape(x):

    data = get_data("{}/{}.json".format(training_path, x))

    len_train = len(data['train'])

    ans = True

    for i in range(len_train):

        input_array = np.array(data['train'][i]['input'])

        output_array = np.array(data['train'][i]['output'])

        if input_array.shape!=output_array.shape:

            ans = False

    return ans



df_train['inputshape_is_output_shape'] = -1

df_train['inputshape_is_output_shape'] = df_train['id'].apply(lambda x: get_inputshape_is_output_shape(x))

df_train.head()
def get_inputshape_is_same(x):

    data = get_data("{}/{}.json".format(training_path, x))

    len_train = len(data['train'])

    ans = True

    input_shape = np.array(data['train'][0]['input']).shape

    for i in range(len_train):

        input_array = np.array(data['train'][i]['input'])

        output_array = np.array(data['train'][i]['output'])

        if input_array.shape!=input_shape:

            ans = False

    return ans



df_train['inputshape_is_same'] = -1

df_train['inputshape_is_same'] = df_train['id'].apply(lambda x: get_inputshape_is_same(x))

df_train.head()
def get_outputshape_is_same(x):

    data = get_data("{}/{}.json".format(training_path, x))

    len_train = len(data['train'])

    ans = True

    output_shape = np.array(data['train'][0]['output']).shape

    for i in range(len_train):

        input_array = np.array(data['train'][i]['input'])

        output_array = np.array(data['train'][i]['output'])

        if output_array.shape!=output_shape:

            ans = False

    return ans



df_train['outputshape_is_same'] = -1

df_train['outputshape_is_same'] = df_train['id'].apply(lambda x: get_outputshape_is_same(x))

df_train.head()
def get_inputunique_is_outputunique(x):

    data = get_data("{}/{}.json".format(training_path, x))

    len_train = len(data['train'])

    ans = True

    for i in range(len_train):

        input_array = np.array(data['train'][i]['input'])

        output_array = np.array(data['train'][i]['output'])

        if np.unique(input_array).shape!=np.unique(output_array).shape:

            ans = False

        else:

            for j in range(len(np.unique(input_array))):

                if np.unique(input_array)[j]!=np.unique(output_array)[j]:

                    ans = False

    return ans



df_train['inputunique_is_outputunique'] = -1

df_train['inputunique_is_outputunique'] = df_train['id'].apply(lambda x: get_inputunique_is_outputunique(x))

df_train.head()
def get_inputunique_is_same(x):

    data = get_data("{}/{}.json".format(training_path, x))

    len_train = len(data['train'])

    ans = True

    input_unique = np.unique(np.array(data['train'][0]['input']))

    for i in range(len_train):

        input_array = np.array(data['train'][i]['input'])

        output_array = np.array(data['train'][i]['output'])

        if np.unique(input_array).shape!=input_unique.shape:

            ans = False

        else:

            for j in range(len(input_unique)):

                if np.unique(input_array)[j]!=input_unique[j]:

                    ans = False

    return ans



df_train['inputunique_is_same'] = -1

df_train['inputunique_is_same'] = df_train['id'].apply(lambda x: get_inputunique_is_same(x))

df_train.head()
def get_outputunique_is_same(x):

    data = get_data("{}/{}.json".format(training_path, x))

    len_train = len(data['train'])

    ans = True

    output_unique = np.unique(np.array(data['train'][0]['output']))

    for i in range(len_train):

        input_array = np.array(data['train'][i]['input'])

        output_array = np.array(data['train'][i]['output'])

        if np.unique(output_array).shape!=output_unique.shape:

            ans = False

        else:

            for j in range(len(output_unique)):

                if np.unique(output_array)[j]!=output_unique[j]:

                    ans = False

    return ans



df_train['outputunique_is_same'] = -1

df_train['outputunique_is_same'] = df_train['id'].apply(lambda x: get_outputunique_is_same(x))

df_train.head()
tmp = df_train.iloc[:,3:].values==df_train.iloc[0,3:].values

tmp = tmp.min(axis=1)

tmp
tmp = df_train.iloc[:,3:].values==df_train.iloc[0,3:].values

tmp = tmp.min(axis=1)

df_tmp = df_train[tmp]

df_tmp
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
idx = 0

task = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

plot_task(task)
for idx in df_tmp.index:

    print("{:3d}: {}".format(idx, df_tmp['id'][idx]))

    task = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

    plot_task(task)
task
def plot_task2(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(3, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one(axs[0,i],i,'train','input')

        plot_one(axs[1,i],i,'train','output') 

        plot_one(axs[12,i],i,'train','predict')        

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
import cv2

from skimage.transform import resize



def plot_one2(task, ax, i,train_or_test,input_or_output):

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

    

    



def plot_task2(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(3, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one2(task, axs[0,i],i,'train','input')

        plot_one2(task, axs[1,i],i,'train','output') 

        plot_one2(task, axs[2,i],i,'train','predict')        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(3, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one2(task, axs[0],0,'test','input')

        plot_one2(task, axs[1],0,'test','output')  

        plot_one2(task, axs[2],0,'test','predict')    

    else:

        for i in range(num_test):      

            plot_one2(task, axs[0,i],i,'test','input')

            plot_one2(task, axs[1,i],i,'test','output')  

            plot_one2(task, axs[2,i],i,'test','predict') 

    plt.tight_layout()

    plt.show() 

    

def proc_init(data):

    for i in range(len(data['train'])):

        t_in = data['train'][i]['input']

        t_out = data['train'][i]['output']

#         t_pred = data['train'][i]['predict']

#         data['train'][i]['predict'] = func(t_in, t_out, t_pred)

        data['train'][i]['input'] = np.array(t_in)

        data['train'][i]['output'] = np.array(t_out)

        data['train'][i]['predict'] = np.array(t_in)

    

    for i in range(len(data['test'])):

        t_in = data['test'][i]['input']

        t_out = data['test'][i]['output']

#         t_pred = data['train'][i]['predict']

#         data['train'][i]['predict'] = func(t_in, t_out, t_pred)

        data['test'][i]['input'] = np.array(t_in)

        data['test'][i]['output'] = np.array(t_out)

        data['test'][i]['predict'] = np.array(t_in)

    return data





def proc_base(data, func):

    for i in range(len(data['train'])):

        t_in = data['train'][i]['input']

        t_out = data['train'][i]['output']

        t_pred = data['train'][i]['predict']

        data['train'][i]['predict'] = func(t_in, t_out, t_pred)

    

    for i in range(len(data['test'])):

        t_in = data['test'][i]['input']

        t_out = data['train'][0]['output']

        t_pred = data['test'][i]['predict']

        data['test'][i]['predict'] = func(t_in, t_out, t_pred)

    return data



def func_resize(t_in, t_out, t_pred):

    tmp = np.zeros_like(t_out)

    h, w = t_out.shape[0]//t_in.shape[0], t_out.shape[1]//t_in.shape[1]

    for i in range(t_in.shape[0]):

        for j in range(t_in.shape[1]):

            tmp[i*h:(i+1)*h, j*w:(j+1)*w] = t_pred[i,j]

    return tmp



def func_tile0(t_in, t_out, t_pred):

    tmp = t_pred.copy()

    h, w = t_pred.shape[0]//t_in.shape[0], t_pred.shape[1]//t_in.shape[1]

    

    for i in range(t_in.shape[0]):

        for j in range(t_in.shape[1]):

            if t_in[i, j]==0:

                tmp[i::t_in.shape[0], j::t_in.shape[1]] = 0

    return tmp



idx = 0

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, func_resize)

data_idx = proc_base(data_idx, func_tile0)

# print(data_idx)

plot_task2(data_idx)
def slice4(t_in, t_out, t_pred):

    tmp = t_pred[:t_pred.shape[0]//2, :t_pred.shape[1]//2]

    return tmp



def crop0(t_in, t_out, t_pred):

    tmp = t_pred.copy()

    tmp = tmp[tmp.max(axis=1)!=0]

    tmp = tmp[:,tmp.max(axis=0)!=0]

    return tmp



idx = 38

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, crop0)

data_idx = proc_base(data_idx, slice4)

# print(data_idx)

plot_task2(data_idx)
def copy_horizontal(t_in, t_out, t_pred):

    tmp = np.concatenate([t_pred, t_pred], axis=1)

    return tmp



idx = 56

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, crop0)

data_idx = proc_base(data_idx, copy_horizontal)

# print(data_idx)

plot_task2(data_idx)
def copy_flip(t_in, t_out, t_pred):

    tmp = np.concatenate([t_pred, t_pred[:,::-1]], axis=1)

    return tmp



def copy_flop(t_in, t_out, t_pred):

    tmp = np.concatenate([t_pred, t_pred[::-1]], axis=0)

    return tmp



idx = 82

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, copy_flip)

data_idx = proc_base(data_idx, copy_flop)

# print(data_idx)

plot_task2(data_idx)
def flip(t_in, t_out, t_pred):

    tmp = t_pred[:,::-1]

    return tmp



def flop(t_in, t_out, t_pred):

    tmp = t_pred[::-1]

    return tmp



idx = 115

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, flop)

data_idx = proc_base(data_idx, copy_flop)

# print(data_idx)

plot_task2(data_idx)
idx = 151

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, copy_flip)

data_idx = proc_base(data_idx, copy_flop)

# print(data_idx)

plot_task2(data_idx)
idx = 163

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, copy_flip)

# data_idx = proc_base(data_idx, copy_flop)

# print(data_idx)

plot_task2(data_idx)
idx = 171

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

# data_idx = proc_base(data_idx, copy_flip)

data_idx = proc_base(data_idx, copy_flop)

# print(data_idx)

plot_task2(data_idx)
def copy_rotate90(t_in, t_out, t_pred):

    tmp = np.concatenate([t_pred, t_pred.T[:,::-1]], axis=1)

    return tmp



def rotate90(t_in, t_out, t_pred):

    tmp = t_pred.T[:,::-1]

    return tmp

def rotate270(t_in, t_out, t_pred):

    tmp = t_pred.T[::-1]

    return tmp



def copy_rotate180(t_in, t_out, t_pred):

    tmp = np.concatenate([t_pred, t_pred[::-1,::-1]], axis=1)

    return tmp



idx = 193

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, copy_rotate90)

# data_idx = proc_base(data_idx, flop)

data_idx = proc_base(data_idx, rotate270)

# data_idx = proc_base(data_idx, copy_rotate180)

# print(data_idx)

plot_task2(data_idx)
idx = 222

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, func_resize)

# print(data_idx)

plot_task2(data_idx)
def proc_base(data, func):

    for i in range(len(data['train'])):

        t_in = data['train'][i]['input']

        t_out = data['train'][i]['output']

        t_pred = data['train'][i]['predict']

        try:

            data['train'][i]['predict'] = func(t_in, t_out, t_pred)

        except:

            data['train'][i]['predict'] = t_pred

    for i in range(len(data['test'])):

        t_in = data['test'][i]['input']

        t_out = data['train'][0]['output']

        t_pred = data['test'][i]['predict']

        try:

            data['test'][i]['predict'] = func(t_in, t_out, t_pred)

        except:

            data['test'][i]['predict'] = t_pred

    return data





def tile_reflect(t_in, t_out, t_pred):

    tmp = np.zeros_like(t_out)

    h, w = t_out.shape[0]//t_in.shape[0], t_out.shape[1]//t_in.shape[1]

    

    for i in range(h):

        for j in range(w):

            tile = t_pred

            if i%2==1: tile = tile[::-1]

            if j%2==1: tile = tile[:, ::-1]

            tmp[i*t_in.shape[0]:(i+1)*t_in.shape[0], j*t_in.shape[1]:(j+1)*t_in.shape[1]] = tile

            

    return tmp

def rotate180(t_in, t_out, t_pred):

    tmp = t_pred[::-1, ::-1]

    return tmp





idx = 210

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

data_idx = proc_base(data_idx, rotate180)

data_idx = proc_base(data_idx, tile_reflect)

# print(data_idx)

plot_task2(data_idx)
func_list = [

    func_resize,

    func_tile0,

    slice4,

    crop0,

    copy_horizontal,

    copy_flip,

    copy_flop,

    flip,

    flop,

    copy_rotate90,

    rotate90,

    rotate270,

    copy_rotate180,

    tile_reflect,

    rotate180,  

]
def judge(data):

    ans = True

    for i in range(len(data['train'])):

        t_in = data['train'][i]['input']

        t_out = data['train'][i]['output']

        t_pred = data['train'][i]['predict']

        if t_out.shape!=t_pred.shape: ans = False

        elif (t_out==t_pred).min()==0: ans = False

    return ans





idx = 210

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

print(judge(data_idx))

data_idx = proc_base(data_idx, rotate180)

data_idx = proc_base(data_idx, tile_reflect)



print(judge(data_idx))

# print(data_idx)

plot_task2(data_idx)
import copy
idx = 0

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

for i in range(1000):

    data_tmp = copy.deepcopy(data_idx)

    func_tmp = np.random.choice(func_list, 2)

    for func in func_tmp:

        data_tmp = proc_base(data_tmp, func)

    if judge(data_tmp):

        print(i, "Find!")

        break

plot_task2(data_tmp)
idx = 0

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

try_list = [(1, len(func_list)), (2, len(func_list)**2), (3, len(func_list)**3)]

for num_func, num_try in try_list:

    for i in range(num_try):

        data_tmp = copy.deepcopy(data_idx)

        func_tmp = np.random.choice(func_list, num_func)

        for func in func_tmp:

            data_tmp = proc_base(data_tmp, func)

        if judge(data_tmp):

            print(i, "Find!")

            break

plot_task2(data_tmp)
idx = np.random.randint(len(df_train))

print(idx)

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

try_list = [(1, len(func_list)), (2, len(func_list)**2), (3, len(func_list)**3)]

for num_func, num_try in try_list:

    for i in range(num_try):

        data_tmp = copy.deepcopy(data_idx)

        func_tmp = np.random.choice(func_list, num_func)

        for func in func_tmp:

            data_tmp = proc_base(data_tmp, func)

        if judge(data_tmp):

            print(i, "Find!")

            break

print(func_tmp)

plot_task2(data_tmp)
import copy
def proc_init(data):

    for i in range(len(data['train'])):

        t_in = data['train'][i]['input']

        t_out = data['train'][i]['output']

#         t_pred = data['train'][i]['predict']

#         data['train'][i]['predict'] = func(t_in, t_out, t_pred)

        data['train'][i]['input'] = np.array(t_in)

        data['train'][i]['output'] = np.array(t_out)

        data['train'][i]['predict'] = np.array(t_in)

    rate_h = copy.deepcopy(data['train'][i]['output'].shape[0]/data['train'][i]['input'].shape[0])

    rate_w = copy.deepcopy(data['train'][i]['output'].shape[1]/data['train'][i]['input'].shape[1])

    for i in range(len(data['test'])):

        t_in = np.array(data['test'][i]['input'])

        t_out = np.zeros([int(t_in.shape[0]*rate_h),int(t_in.shape[1]*rate_w)], dtype=t_in.dtype)

#         t_pred = data['train'][i]['predict']

#         data['train'][i]['predict'] = func(t_in, t_out, t_pred)

        data['test'][i]['input'] = t_in

        data['test'][i]['output'] = t_out

        data['test'][i]['predict'] = np.array(t_in)

    return data



count = 0

for idx in range(len(df_train.iloc[:])):

    print(idx)

    data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

    data_idx = proc_init(data_idx)

    try_list = [(1, len(func_list)), (2, len(func_list)**2), 

#                 (3, len(func_list)**3), (4, len(func_list)**3)

               ]

    flag = False

#     plot_task2(data_idx)

    for num_func, num_try in try_list:

        for i in range(num_try):

            data_tmp = copy.deepcopy(data_idx)

            func_tmp = np.random.choice(func_list, num_func)

            for func in func_tmp:

                data_tmp = proc_base(data_tmp, func)

            if judge(data_tmp):

                print(i, "Find!")

                print(func_tmp)

                plot_task2(data_tmp)

                flag = True

                count += 1

                break

        if flag: break

print(count)
def flattener(pred):

    str_pred = str([row for row in pred.tolist()])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
idx = 0

data_idx = get_data("{}/{}.json".format(training_path, df_train['id'][idx]))

data_idx = proc_init(data_idx)

try_list = [(1, len(func_list)), (2, len(func_list)**2), (3, len(func_list)**3)]

flag = False

for num_func, num_try in try_list:

    for i in range(num_try):

        data_tmp = copy.deepcopy(data_idx)

        func_tmp = np.random.choice(func_list, num_func)

        for func in func_tmp:

            data_tmp = proc_base(data_tmp, func)

        if judge(data_tmp):

            print(i, "Find!")

            flag = True

            break

        if flag: break

    if flag: break

print(data_tmp['test'][0]['predict'])

print(flattener(data_tmp['test'][0]['predict']))

plot_task2(data_tmp)
df_sub = pd.read_csv("../input/abstraction-and-reasoning-challenge/sample_submission.csv")

df_sub.head()
df_test = pd.DataFrame(test_tasks, columns=['id'])

df_test['id'] = df_test['id'].apply(lambda x: x[:-5])

df_test.head()
def record_answer(idx, df_sub, data):

    for i in range(len(data['test'])):

        t_in = data['test'][i]['input']

        t_out = data['test'][i]['output']

        t_pred = data['test'][i]['predict']

        ans = flattener(t_pred)

        df_sub['output'][df_sub['output_id']=="{}_{}".format(idx, i)] = ans

#         print("{}_{}".format(idx, i), ans)

    return df_sub
count = 0

for idx in range(len(df_test.iloc[:])):

    print(idx)

    data_idx = get_data("{}/{}.json".format(test_path, df_test['id'][idx]))

    data_idx = proc_init(data_idx)

    try_list = [(1, len(func_list)*3), (2, len(func_list)**2*3), 

#                 (3, len(func_list)**3), 

#                 (4, len(func_list)**3)

               ]

    flag = False

#     plot_task2(data_idx)

    for num_func, num_try in try_list:

        for i in range(num_try):

            data_tmp = copy.deepcopy(data_idx)

            func_tmp = np.random.choice(func_list, num_func)

            for func in func_tmp:

                data_tmp = proc_base(data_tmp, func)

            if judge(data_tmp):

                print(i, "Find!")

                print(func_tmp)

                plot_task2(data_tmp)

                flag = True

                count += 1

                df_sub = record_answer(df_test['id'][idx], df_sub, data_tmp)

                break

        if flag: break

    if flag==False:

        df_sub = record_answer(df_test['id'][idx], df_sub, data_idx)

print(count)
df_sub.to_csv("submission.csv", index=False)
df_sub.head(20)