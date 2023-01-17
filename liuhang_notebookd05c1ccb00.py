import numpy as np

import pandas as pd

import sklearn.linear_model as lm

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')