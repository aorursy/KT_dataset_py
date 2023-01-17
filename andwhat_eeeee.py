import numpy as np
import pandas as pd
from pandas import DataFrame as df
from pandas import read_csv

import sklearn

import matplotlib.pyplot as pl
%matplotlib inline

import itertools
from sklearn.metrics import roc_auc_score

train = read_csv('train11.txt', names=['Word', 'Label'])

from sklearn.feature_extraction.text import CountVectorizer

test = read_csv('linear_test.txt', names=['Word'])
