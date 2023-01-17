# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import json

import numpy as np

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

all_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if 'json' in filename: 

            all_files.append(os.path.join(dirname, filename))

    

def load_json(path):

    with open(path) as f:

        return json.load(f)

    

loaded = [load_json(x) for x in all_files]

    

#print(loaded)



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
loaded[0]
test1 = np.array(loaded[0]['train'][0]['input'])
len(loaded[0]['test'])
def score_rand_dist(arr):

    return distance_matrix(arr,arr).sum()



def find_n_rand_with_max_dist(limit,count,dimensions):

    best = np.random.rand(count,dimensions)

    best = (best, score_rand_dist(best))

    worst = (best[0].copy(), best[1])

    for _ in range(limit):

        next_test = np.random.rand(count,dimensions)

        c_score = score_rand_dist(next_test)

        if (c_score > best[1]):

            #print('new best: ' + str(c_score))

            best = (next_test, c_score)

        elif (c_score < worst[1]):

            #print('new worst: ' + str(c_score))

            worst = (next_test, c_score)

    return (best, worst)

        
b, w = find_n_rand_with_max_dist(180000, 10, 3)

plt.imshow(np.expand_dims(b[0],0))
final_cols = np.array([

        [0.0,0.0,0.0],

        [0.0,0.4,0.1],

        [0.8,0.7,0.7],

        [0.1,0.2,0.7],

        [0.9,0.2,0.1],

        [0.5,0.9,0.1],

        [0.8,0.6,0.1],

        [0.2,0.8,0.9],

        [0.7,0.1,0.7],

        [0.9,0.9,0.0]]) #b[0]
plt.imshow(np.expand_dims(final_cols,0))
def to_rgb(arr):

    idx_to_col = {idx:col for idx,col in zip(range(len(final_cols)),final_cols)}

    c_arr = np.zeros((arr.shape[0],arr.shape[1], 3))

    for x,column in enumerate(arr):

        for y,item in enumerate(column):

            c_arr[x][y] = idx_to_col[item]

    return c_arr
prob_no = 405#205

mode = 'train'

#it_no = 0

p_count = len(loaded[prob_no][mode])

fig, axes = plt.subplots(2,p_count)

for it_no in range(p_count):

    axes[0,it_no].axis('off')

    axes[0,it_no].imshow(to_rgb(np.array(loaded[prob_no][mode][it_no]['input'])), interpolation='nearest')

    axes[1,it_no].axis('off')

    axes[1,it_no].imshow(to_rgb(np.array(loaded[prob_no][mode][it_no]['output'])), interpolation='nearest')
in_out_dim_match_probs = []

for idx,prob in enumerate(loaded):

    dims_match = True

    for train_prob in prob['train']:

        in_arr = np.array(train_prob['input'])

        out_arr = np.array(train_prob['output'])

        if (in_arr.shape != out_arr.shape):

            dims_match = False

    if dims_match:

        in_out_dim_match_probs.append(idx)
in_out_dim_match_probs