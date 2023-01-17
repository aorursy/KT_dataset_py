# load libraries



import pandas as pd # Pandas is an easy-to-use data structures and data analysis tools

import numpy as np # NumPy is the fundamental package for scientific computing

import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library
# load dataset

dataset = pd.read_csv("../input/train.csv")



# shape

print(dataset.shape)
# types

print(dataset.dtypes)
# head

print (dataset.head(5))
# descriptions

pd.set_option('precision', 1)

print(dataset.describe())
#correlation

pd.set_option('precision', 2)

print(dataset.corr(method='pearson'))