# -*- coding: utf-8 -*-

"""

Thanks to tinrtgu for the wonderful base script

Use pypy for faster computations.!

"""

import csv

from datetime import datetime

from csv import DictReader

from math import exp, log, sqrt





# TL; DR, the main training process starts on line: 250,

# you may want to start reading the code from there





##############################################################################

# parameters #################################################################

##############################################################################



# A, paths

data_path = "../input/"

train = data_path+'clicks_train.csv'               # path to training file

test = data_path+'clicks_test.csv'                 # path to testing file

submission = 'sub_proba.csv'  # path of to be outputted submission file



# B, model

alpha = .1  # learning rate

beta = 0.   # smoothing parameter for adaptive learning rate

L1 = 0.    # L1 regularization, larger value means more regularized

L2 = 0.     # L2 regularization, larger value means more regularized



# C, feature/hash trick

D = 2 ** 20             # number of weights to use

interaction = False     # whether to enable poly2 feature interactions



# D, training/validation

epoch = 1       # learn training data for N passes

holdafter = None   # data after date N (exclusive) are used as validation

holdout = None  # use every N training instance for holdout validation