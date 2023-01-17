import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.dtypes
train.head()