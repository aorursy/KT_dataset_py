# 5 Day Data Challenge, Day 3

# Use a t-test



import matplotlib.pyplot as plt

import matplotlib.patches as patches

import pandas as pd

from scipy import stats

import pylab 



# read movies.csv 

file = '../input/movies.csv'

data = pd.read_csv(file, encoding='latin1')



data.head()
# check normality of data; most of the samples should be plotted along the diagonal

stats.probplot(data['score'], dist='norm', plot=pylab)
# comedy movies

comedy = data[data['genre']=='Comedy']



# action movies

action = data[data['genre']=='Action']
# is there a meaningful difference between the score of comedy and action movies?

stats.ttest_ind(comedy['score'], action['score'])



# the p-value (probability) is 0.08 so there's a minor difference, but it shouldn't be relevant.
# plot histograms

plt.hist(comedy['score'])

plt.hist(action['score'])



# title

plt.title('Score of comedy/action movies')



# legends

blue_patch = patches.Patch(label='Comedy')

or_patch = patches.Patch(color='orange', label='Action')

plt.legend(handles=[blue_patch, or_patch])
