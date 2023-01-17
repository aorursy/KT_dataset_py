# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import libraries necessary for this project

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from matplotlib import pyplot as plt



# Load the Boston housing dataset

data = pd.read_csv('../input/bostonhoustingmlnd/housing.csv')

prices = data['MEDV']

features = data.drop('MEDV', axis = 1)
fig, axs = plt.subplots(ncols=2, figsize=(14,6))



for i in range(len(axs)):

    axs[i].set_ylim([0, 1200000])

# RM vs Price

slope, intercept, r_value, p_value, std_err = stats.linregress(data['RM'],data['MEDV'])

_ = sns.regplot(x=features['RM'], y=prices, ax=axs[0], line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

_.set_ylabel('House Price')

_.legend()



# LSSAT vs Price

slope, intercept, r_value, p_value, std_err = stats.linregress(data['LSTAT'],data['MEDV'])

_ = sns.regplot(x=features['LSTAT'], y=prices, ax=axs[1], line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

_.set_ylabel('House Price')

_.legend()



#plot

plt.show()