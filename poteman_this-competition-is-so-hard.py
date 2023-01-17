import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np
# input_path = "../input/abstraction-and-reasoning-challenge/"

input_path = "/kaggle/input/abstraction-and-reasoning-challenge"



for dirname, _, filenames in os.walk(input_path):

    print(dirname)
from pathlib import Path



data_path = Path(input_path)

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))

print(training_tasks[:3])
task_file = str(training_path / '00d62c1b.json')



with open(task_file, 'r') as f:

    task = json.load(f)



print(task.keys())
n_train_pairs = len(task['train'])

n_test_pairs = len(task['test'])



print(f'task contains {n_train_pairs} training pairs')

print(f'task contains {n_test_pairs} test pairs')
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

    

def plot_task_sub(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    fig, axs = plt.subplots(1, 3, figsize=(15,15))

    axs[0].imshow(task['train'][0]['input'], cmap=cmap, norm=norm)

    axs[0].axis('off')

    axs[0].set_title('Train Input')

    axs[1].imshow(task['train'][0]['output'], cmap=cmap, norm=norm)

    axs[1].axis('off')

    axs[1].set_title('Train Output')

    axs[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)

    axs[2].axis('off')

    axs[2].set_title('Test Input')

    plt.tight_layout()

    plt.show()

    

def plot_task_item(task, idx):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    fig, axs = plt.subplots(1, 3, figsize=(15,15))

    axs[0].imshow(task['train'][idx]['input'], cmap=cmap, norm=norm)

    axs[0].axis('off')

    axs[0].set_title('Train Input')

    axs[1].imshow(task['train'][idx]['output'], cmap=cmap, norm=norm)

    axs[1].axis('off')

    axs[1].set_title('Train Output')

    axs[2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)

    axs[2].axis('off')

    axs[2].set_title('Test Input')

    plt.tight_layout()

    plt.show()
plot_task(task)
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
def read_json_file(fileName):

    with open(fileName, 'r') as f:

        return json.load(f) 
submission.shape
def task_train001(df_input):

    

    df_input = np.array(df_input)

    df_output = np.array(df_output)



    color_input, color_output = input_all_colors[0], new_colors[0]



    def get_closed_area(arr):

        # depth first search

        H, W = arr.shape

        Dy = [0, -1, 0, 1]

        Dx = [1, 0, -1, 0]

        arr_padded = np.pad(arr, ((1, 1), (1, 1)), "constant", constant_values=0)

        searched = np.zeros(arr_padded.shape, dtype=bool)

        searched[0, 0] = True

        q = [(0, 0)]

        while q:

            y, x = q.pop()

            for dy, dx in zip(Dy, Dx):

                y_, x_ = y + dy, x + dx

                if not 0 <= y_ < H + 2 or not 0 <= x_ < W + 2:

                    continue

                if not searched[y_][x_] and arr_padded[y_][x_] == 0:

                    q.append((y_, x_))

                    searched[y_, x_] = True

        res = searched[1:-1, 1:-1]

        res |= arr == color_input

        return ~res



    output = df_input.copy()

    output[get_closed_area(df_input)] = color_output

    

    return output



def wrapper_task_train001(df_train, df_test_item):



    # 新增一种颜色

    for df in df_train:

        input_all_colors = [x for x in list(np.unique(df['input'])) if x != 0]

        output_all_colors = [x for x in list(np.unique(df['output'])) if x != 0]

        new_colors = [x for x in output_all_colors if x not in input_all_colors]

        assert (len(new_colors) == 1)

    #         print(input_all_colors, output_all_colors)



    # 判断是否要删除边框

    remove_border = False

    if len(output_all_colors) - len(input_all_colors) == 0:

        remove_border = True



    def task_train001(x):

        x = np.array(x)

        green, yellow = input_all_colors[0], new_colors[0]



        def get_closed_area(arr):

            # depth first search

            H, W = arr.shape

            Dy = [0, -1, 0, 1]

            Dx = [1, 0, -1, 0]

            arr_padded = np.pad(arr, ((1, 1), (1, 1)), "constant", constant_values=0)

            searched = np.zeros(arr_padded.shape, dtype=bool)

            searched[0, 0] = True

            q = [(0, 0)]

            while q:

                y, x = q.pop()

                for dy, dx in zip(Dy, Dx):

                    y_, x_ = y + dy, x + dx

                    if not 0 <= y_ < H + 2 or not 0 <= x_ < W + 2:

                        continue

                    if not searched[y_][x_] and arr_padded[y_][x_] == 0:

                        q.append((y_, x_))

                        searched[y_, x_] = True

            res = searched[1:-1, 1:-1]

            res |= arr == green

            return ~res



        y = x.copy()

        y[get_closed_area(x)] = yellow



        if remove_border:

            y[y == input_all_colors[0]] = 0



        return y



    for df in df_train:

        assert (np.array_equal(task_train001(df['input']), df['output']) == True)

    print(np.array_equal(task_train001(df_test_item['input']), df_test_item['output']))



    return task_train001(df_test_item['input'])
from tqdm import tqdm, tqdm_notebook 



for output_id in tqdm_notebook(['00d62c1b_0', 'd5d6de2d_0', 'a5313dff_0'], total = 3):



#     print(output_id)

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    

    fileName = str(training_path / str(task_id + '.json'))



    task = read_json_file(fileName)

    plot_task_sub(task)



    try:

        res = wrapper_task_train001(task['train'], task['test'][pair_id])

        pred_1 = flattener(res.tolist())

        print("wrapper_task_train001 success", pred_1)

    except:

        data = task['test'][pair_id]['input']

        pred_1 = flattener(data)



    # concatenate and add to the submission output

    pred = pred_1

    

    submission.loc[output_id, 'output'] = pred



submission.to_csv('submission.csv')
submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')

used_k = len(submission)



for output_id in tqdm_notebook(submission.index[:used_k], total = len(submission.index[:used_k])):

# for output_id in tqdm_notebook(['00d62c1b_0', 'd5d6de2d_0', 'a5313dff_0'], total = 3):



#     print(output_id)

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    

    fileName = str(test_path / str(task_id + '.json'))

#     fileName = str(training_path / str(task_id + '.json'))



    task = read_json_file(fileName)

#     plot_task_sub(task)



    try:

        res = wrapper_task_train001(task['train'], task['test'][pair_id])

        pred_1 = flattener(res.tolist())

        print("wrapper_task_train001 success", pred_1)

    except:

        data = task['test'][pair_id]['input']

        pred_1 = flattener(data)



    # concatenate and add to the submission output

    pred = pred_1

    

    submission.loc[output_id, 'output'] = pred



submission.to_csv('submission.csv')