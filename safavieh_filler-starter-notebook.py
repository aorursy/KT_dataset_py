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

print(training_tasks[:3])
task_file = str(training_path / '00d62c1b.json')



with open(task_file, 'r') as f:

    task = json.load(f)



print(task.keys())
n_train_pairs = len(task['train'])

n_test_pairs = len(task['test'])



print(f'task contains {n_train_pairs} training pairs')

print(f'task contains {n_test_pairs} test pairs')
display(task['train'][0]['input'])

display(task['train'][0]['output'])
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
plot_task(task)
import scipy.ndimage.morphology as mp



cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

fig, axs = plt.subplots(1, 12, figsize=(15,5))





def findHoles(im):

    Filled = mp.binary_fill_holes(im)

    Holes = Filled.astype(int) - (im > 0).astype(int)

    return Holes



im = np.array(task['train'][0]['input'])

out = np.array(task['train'][0]['output'])

Holes = findHoles(im)

axs[0].imshow(im, cmap=cmap, norm=norm)

axs[0].set_title('input')

axs[1].imshow(out, cmap=cmap, norm=norm)

axs[1].set_title('output')



for FillValue in range(10):

    Filled = np.where(Holes, FillValue, im)

    axs[FillValue+2].imshow(Filled, cmap=cmap, norm=norm)

    axs[FillValue+2].set_title(FillValue)

    if (Filled==out).all():

        print (FillValue)
def getFillValue(im, out, maxValue=10):

    Holes = findHoles(im)

    for FillValue in range(maxValue+1):

        Filled = np.where(Holes, FillValue, im)

        if (Filled==out).all():

            return FillValue

    return 0
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
def getDefaultPred(task, pair_id):

    data = task['test'][pair_id]['input'] # test pair input

    # for the first guess, predict that output is unchanged

    pred_1 = flattener(data)

    # for the second guess, change all 0s to 5s

    data = [[5 if i==0 else i for i in j] for j in data]

    pred_2 = flattener(data)

    # for the last gues, change everything to 0

    data = [[0 for i in j] for j in data]

    pred_3 = flattener(data)

    # concatenate and add to the submission output

    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 

    return pred



FillerCount = 0

for output_id in submission.index:

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:

        task = json.load(read_file)

    if all([np.array(tr['input']).shape == np.array(tr['output']).shape for tr in task['train']]):

        FillValues = np.unique([getFillValue(np.array(tr['input']), np.array(tr['output'])) for tr in task['train']])

        if len(FillValues)==1 and FillValues[0] > 0:

            print('Found a Filler')

            FillerCount += 1

            FillValue = FillValues[0]

            im = np.array(task['test'][pair_id]['input'])

            Holes = findHoles(im)

            Filled = np.where(Holes, FillValue, im).tolist()

            pred = flattener(Filled)

        else:

            pred = getDefaultPred(task, pair_id)

    else:

        pred = getDefaultPred(task, pair_id)

    

    submission.loc[output_id, 'output'] = pred



submission.to_csv('submission.csv')

print(f'{FillerCount} Fillers found in total')