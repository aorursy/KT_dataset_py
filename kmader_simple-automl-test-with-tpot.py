import os

import pandas as pd

import numpy as np

from tpot import TPOTRegressor

tpot_settings = dict(verbosity=2, random_state = 1234, scoring = 'mean_absolute_error', warm_start = True)

REG_COLUMN = 'price_doc'

TINY_TEST = True
macro_df = pd.read_csv(os.path.join('..', 'input', 'macro.csv'))

def read_house_df(fname):

    in_df = pd.read_csv(os.path.join('..', 'input', fname))

    if TINY_TEST:

        return in_df

    return in_df.merge(macro_df, on = 'timestamp')



train_df = read_house_df('train.csv')

test_df = read_house_df('test.csv')

print('Training set loaded:', train_df.shape)

train_df.sample(3)
# get an idea of the variable types

train_df.apply(lambda x: type(x.values[0]),0)
%%time

auto_reg = TPOTRegressor(generations=2, population_size=5, **tpot_settings)

if TINY_TEST:

    auto_reg = TPOTRegressor(generations=1, population_size=3, **tpot_settings)

X_df = train_df[[ccol for ccol in train_df.columns if (ccol not in ['id'])]].select_dtypes(include=[np.number]).dropna(1)

if TINY_TEST:

    X_df = X_df.sample(5000)

print('fitting:', X_df.shape)

auto_reg.fit(X_df.drop(REG_COLUMN,1), X_df[REG_COLUMN])
test_feat_df = test_df[['id']+list(X_df.drop(REG_COLUMN,1).columns)].dropna(0)

guess_df = test_feat_df[['id']]

print('Test size', test_df.shape, '->', test_feat_df.shape)

guess_df[REG_COLUMN] = auto_reg.predict(test_feat_df.drop('id',1))

guess_df.sample(3)
guess_df.to_csv('guess.csv', index = False)