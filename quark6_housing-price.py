# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#import matplotlib.pyplot as pyplot

import matplotlib

import matplotlib.pyplot

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

#from matplotlib import cm

from sklearn import preprocessing, manifold, linear_model, metrics, model_selection, ensemble

import seaborn as sns



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

prices = pd.DataFrame({"Number of houses sold at a certain price":train["SalePrice"]})

prices.hist(bins=100)

plt.xlabel('Price')
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

prices = pd.DataFrame({"Number of houses sold at a certain price":train["YearBuilt"]})

prices.hist(bins=100)

plt.xlabel('Price')
matplotlib.pyplot.plot(train["YearBuilt"], train["SalePrice"], 'ro')

#plt.axis([0, 6, 0, 20])

matplotlib.pyplot.show()