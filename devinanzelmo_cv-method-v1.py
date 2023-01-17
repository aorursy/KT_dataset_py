import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import json



import os
# NOTE something to create an more skewed split could be created by not splitting uniformly. 



def split_one_class_uniform(groups, folds, n_folds=3):

    """ Generates folds for one class

    """

    if len(folds) != n_folds:

        raise ValueError('number of folds provided does not match n_folds argument')

        

    # shuffle groups

    random.shuffle(groups)

    while True:

        if len(groups) <= 0:

            break

            

        # there are enough groups left to add one to each fold

        elif len(groups) >= n_folds:

            for i in range(n_folds):

                folds[i].append(groups.pop())

        # we have more then 0 groups, but not enough to add to each fold

        # randomly assign remaining groups to folds

        else:

            # select random fold to contain remaining

            rand_nums = random.sample(range(len(groups)), len(groups)) 

            for i in rand_nums:

                folds[i].append(groups.pop())

    return folds



def create_cv_split(y_tr, n_folds=4):

    """ splits data in n_folds for cv

    """

    group_dict =  dict()

    for (gid, g) in y_tr.groupby(['surface']):

        group_dict[gid] = g['group_id'].unique().tolist()



    folds = [[] for x in range(n_folds)]



    for key, value in group_dict.items():

        folds = split_one_class_uniform(value, folds, n_folds)

        

    return folds



def load_data():

    x_tr = pd.read_csv('../input/X_train.csv', index_col=0)

    x_ts = pd.read_csv('../input/X_test.csv', index_col=0)

    y_tr = pd.read_csv('../input/y_train.csv', index_col=0)



    # remove hard_tile from data 

    # TODO something more intelligent then this.

    # reasoning here: there is a lot of things to improve besides differentiating these classes

    # ignore extra class initially and be ok with 0.06 auto penelty on leaderboard

    # later try to differentiate hard_tile, from hard_tiles_large_space

    y_tr['surface'] = y_tr['surface'].replace('hard_tiles', 'hard_tiles_large_space')

    return x_tr, x_ts, y_tr

x_tr, x_ts, y_tr = load_data()



# lets create 5 random splits, and keep them for later

# using same splits allows for consistant comparison between models

# currently not setting seed so these are different each time this is run. 

for i in range(5):

    folds = create_cv_split(y_tr)

    with open('splits{}.json'.format(i), 'w') as f:

        json.dump(folds, f)