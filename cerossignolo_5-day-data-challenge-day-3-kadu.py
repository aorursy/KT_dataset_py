# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Provides a MATLAB-like plotting framework.

import seaborn as sns # visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

from scipy import stats # ecosystem of open-source software for mathematics, science, and engineering





# lets bring the dataset from a csv file to our pandas dataframe variable

wspd = pd.read_csv('../input/7210_1.csv',delimiter=',')



# the analisys here is to observe if the price of black and brown shoes reveal the same way of princing.

# first we are going to separate the min price of black and brown shoes

blackShoesMinPrice = wspd[wspd['colors']=='Black']['prices.amountMin'] # here we select the color Black of the shoes and stored the prices in a pandas dataframe

brownShoesMinPrice = wspd[wspd['colors']=='Brown']['prices.amountMin'] # the same as above to the brown shoes



# lest explore the results of the mean price of each shoe color using the Student's test,

# t-test and we will see how significant is this diffrence

# we will see two values t-value and the p-value

# high t-value means very different group 

# p values is the probability of the result of the samples ocurred by chance. low values

# means low probability of occurence by chance and this is good



print(stats.ttest_ind(blackShoesMinPrice,brownShoesMinPrice))



# one way to see what the result above shows is to print the hist in the same plot and observe

print(blackShoesMinPrice.hist())

print(brownShoesMinPrice.hist())



# another way is to plt with more information of the distribution

# Set up the matplotlib figure

f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)



# Plot a simple histogram with binsize determined automatically

sns.distplot(blackShoesMinPrice, kde=True, hist=True, rug=True, color="b", ax=axes[0])



# Plot a kernel density estimate and rug plot

sns.distplot(brownShoesMinPrice, kde=True, hist=True, rug=True, color="r", ax=axes[1])


