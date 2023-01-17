import numpy as np

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    

    if len(filenames) > 2:

        print('# of files in {}:'.format(dirname), len(filenames))

        for i in range(5):

            print(os.path.join(dirname, filenames[i]))
from pathlib import Path

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))
import json



# the data augmentation function below takes in one task at a time

task_file_0 = str(training_path / training_tasks[0])



with open(task_file_0, 'r') as f:

    task_0 = json.load(f)



"""

The structure of each loaded task file is,

    {

    'test': [{'input': list, 'output': list}],

    'train': [{'input': list, 'output': list}, {'input': list, 'output': list}, ...]

    }

"""
import itertools



def Data_Aug(task):

    """

    takes in a whole task dictionary

    returns an augmented train set of a task

    [{'input': numpy array, 'output': numpy array}, {'input': numpy array, 'output': numpy array}, ...]

    """

    

    train_task = task['train']

    

    #[1] find out "key colors" used commonly across all the train input/output pairs, which must play an important role on the task rule.

    #Key colors include the background color "black". I suppose it and key colors never suddenly change on the test.

    used_colors_in_each_io_pair = []

    for i, train_io_pair in enumerate(train_task):

        ipt, opt = np.array(train_io_pair['input']), np.array(train_io_pair['output'])

        used_colors_in_each_io_pair.append(np.unique(np.concatenate([ipt.ravel(), opt.ravel()])))

    

    key_colors = np.array([])

    for i in range(len(used_colors_in_each_io_pair)):

        if i == 0:

            key_colors = used_colors_in_each_io_pair[i]

        else:

            key_colors = np.intersect1d(key_colors, used_colors_in_each_io_pair[i])

    #[1] ends

    

    #[2] change "non-key colors" of each train input/output pairs to many color pairs and save the newly-colored input/output pairs.

    new_train_task = []

    for train_io_pair in train_task:

        ipt, opt = np.array(train_io_pair['input']), np.array(train_io_pair['output'])

        

        non_key_colors = [i for i in np.unique(np.concatenate([ipt.ravel(), opt.ravel()])) if i not in key_colors]

        color_comb = list(itertools.product([i for i in range(10) if i not in key_colors], repeat=len(non_key_colors)))

        trns_lst = [tpl for tpl in color_comb if len(np.unique(tpl)) == len(tpl)]

        

        # I assume 100 records are enough for no reason...

        if len(trns_lst) < 100:

            idx_arr = np.random.choice(len(trns_lst), len(trns_lst), replace=False)

        else:

            idx_arr = np.random.choice(len(trns_lst), 100, replace=False)

        

        trns_lst = np.array(trns_lst)[idx_arr]

        

        for i in range(len(trns_lst)):

            new_input = ipt.copy()

            new_output = opt.copy()

            for before, after in zip(non_key_colors, trns_lst[i]):

                new_input[ipt == before] = after

                new_output[opt == before] = after

                

            new_io_pair = {'input': new_input, 'output': new_output}

            new_train_task.append(new_io_pair)

    #[2] ends

        

    return new_train_task
import matplotlib.pyplot as plt

from matplotlib import colors





def plot_one(io_pairs, ax, i, input_or_output):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    input_matrix = io_pairs[i][input_or_output]

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)

    ax.set_yticks([x+0.5 for x in range(len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(input_or_output)





def plot_io_pairs(io_pairs):

    """

    visualizes all the input/output-pair-dictionaries in a list

    takes in [{'input': numpy array, 'output': numpy array}, {'input': numpy array, 'output': numpy array}, ...]

    """

    

    num_pairs = len(io_pairs)

    fig, axs = plt.subplots(2, num_pairs, figsize=(3*num_pairs,3*2))

    

    if num_pairs == 1: 

        plot_one(io_pairs,axs[0],0,'input')

        plot_one(io_pairs,axs[1],0,'output')

    

    else:

        for i in range(num_pairs):

            plot_one(io_pairs,axs[0,i],i,'input')

            plot_one(io_pairs,axs[1,i],i,'output')

        plt.tight_layout()

        plt.show()

from tqdm import tqdm

for i in range(15):

    print("task", i)

    task_file = str(training_path / training_tasks[i])



    with open(task_file, 'r') as f:

        task = json.load(f)

    

    train_io_pairs = Data_Aug(task)

    

    for j in range(0, len(train_io_pairs), 10):

        plot_io_pairs(train_io_pairs[j : j+10])



    plot_io_pairs(task['test'])