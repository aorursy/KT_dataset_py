import numpy as np

import pandas as pd

import json

import math

from pathlib import Path
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

test_path = data_path / 'test'

submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')
# FOR SUBMISSION

def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
def downsample(x,n):

    x0,x1 = math.ceil(x.shape[0]/n), math.ceil(x.shape[1]/n)

    y = np.zeros((x0,x1))

    for i in range(x0):

        for j in range(x1):

            y[i,j] = x[i*n,j*n]

    return y.astype(int)
for output_id in submission.index:

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:

        task = json.load(read_file)

        

        x = np.array(task['test'][pair_id]['input'])

        

        y1 = flattener(downsample(x,2).tolist())

        y2 = flattener(downsample(x,3).tolist())

        y3 = flattener(downsample(x,4).tolist())

        

        submission.loc[output_id, 'output'] = y1 + ' ' + y2 + ' ' + y3

        

submission.to_csv('submission.csv')