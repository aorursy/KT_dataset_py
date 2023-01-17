# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# train_datapath = "titanic_train_dataset.csv"

# test_datapath = "titanic_test_dataset.csv"



train_datapath = "../input/titanic/train.csv"

test_datapath = "../input/titanic/test.csv"

train_data = pd.read_csv(train_datapath, index_col=0)

test_data = pd.read_csv(test_datapath, index_col=0)

train_data.head()
test_data.tail()