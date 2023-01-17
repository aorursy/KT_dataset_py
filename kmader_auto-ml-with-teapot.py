import matplotlib.pylab as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import os

REG_COLUMN = 'loss'

TINY_TEST = True

LOG_REG = True

INCLUDE_CAT_VARS = True # include the categorical variables
train_df = pd.read_csv('../input/train.csv')

if LOG_REG:

    # take the log10 of the value

    train_df[REG_COLUMN] = train_df[REG_COLUMN].map(np.log10)

print('Training Data Shape', train_df.shape)

train_df.sample(3)
from tpot import TPOTRegressor

tpot_settings = dict(verbosity=2, random_state = 1234, scoring = 'mean_absolute_error', warm_start = True)

auto_reg = TPOTRegressor(generations=2, population_size=5, **tpot_settings)

if TINY_TEST:

    auto_reg = TPOTRegressor(generations=1, population_size=3, **tpot_settings)
def make_train_vector(in_df):

    new_df = in_df[[ccol for ccol in in_df.columns if (ccol not in [REG_COLUMN, 'id'])]]

    if INCLUDE_CAT_VARS:

        return new_df

    else:

        return new_df[[ccol for ccol in in_df.columns if ('cat' not in ccol)]]
def make_train_vector(in_df):

    return in_df[[ccol for ccol in in_df.columns if ('cat' not in ccol) and (ccol not in [REG_COLUMN, 'id'])]]
%%time

for i in range(1 if TINY_TEST else 2):

    cur_df = train_df.sample(20000)

    y_train = cur_df[REG_COLUMN]

    x_train = make_train_vector(cur_df)

    auto_reg.fit(x_train, y_train)
test_df = pd.read_csv('../input/test.csv')

test_df.sample(3)
x_test = make_train_vector(test_df)

# we need access to the pipeline to get the probabilities

pred_loss = auto_reg.predict(x_test)

guess_df = test_df[['id']]

guess_df[REG_COLUMN] = np.power(10,pred_loss) if LOG_REG else pred_loss

guess_df.sample(3)
guess_df.to_csv('guess.csv', index = False)