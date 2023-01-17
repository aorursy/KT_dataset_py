import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt #more data visualization

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore') # ignore warnings

from scipy.stats import ttest_ind # for the t-test we'll be doing

from subprocess import check_output 

print(check_output(["ls", "../input"]).decode("utf8"))
cereal = pd.read_csv("../input/cereal.csv")

cereal.head()
hot_cereal = cereal.loc[cereal['type'] == 'H', :] # define a hot_cereal df

cold_cereal = cereal.loc[cereal['type'] == 'C', :] # definte a cold_cereal df
ttest_ind(cold_cereal['sugars'], hot_cereal['sugars'], equal_var = False)
ttest_ind(cold_cereal['calories'], hot_cereal['calories'], equal_var = False)
ax = plt.subplots(figsize=(18,8)) # make our plot larger

# plot the cold cereal sugar distribution

sns.distplot(cold_cereal['sugars'], bins = 10, hist = True,  label = 'cold')

# plot the hot cereal sugar distribution

sns.distplot(hot_cereal['sugars'], bins = 10, hist = True, label = 'hot') 

plt.legend() #show legend
ax = plt.subplots(figsize=(18,8))

sns.distplot(cold_cereal['calories'], bins = 10, hist = True,  label = 'cold')

sns.distplot(hot_cereal['calories'], hist = True, label = 'hot')

plt.legend()