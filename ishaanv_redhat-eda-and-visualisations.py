import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])

test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])

people = pd.read_csv('../input/people.csv', parse_dates=['date'])



df_train = pd.merge(train, people, on='people_id')

df_test = pd.merge(test, people, on='people_id')