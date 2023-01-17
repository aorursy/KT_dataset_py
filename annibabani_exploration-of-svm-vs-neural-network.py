#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import csv

from datetime import datetime

import scipy.stats as stats

from mpl_toolkits.axes_grid1 import make_axes_locatable



#read initial data

card_data = pd.read_csv("../input/creditcard.csv")
#first, let's verify the datatypes in the dataset

print(card_data.info())
#let's also take a look at some basic statistics

data_skew = pd.DataFrame({'mean':card_data.loc[:, 'V1':'V28'].mean(),

                          'std': card_data.loc[:, 'V1':'V28'].std(),

                          'skew':card_data.loc[:, 'V1':'V28'].skew(), 

                          'kurtosis':card_data.loc[:, 'V1':'V28'].kurtosis(),

                          'max':card_data.loc[:, 'V1':'V28'].max(),

                          'min':card_data.loc[:, 'V1':'V28'].min()})

data_skew
#examine variable distribution

plt.rcParams['figure.figsize'] = 16, 16

card_data.loc[:,'V1':'V28'].hist(bins=200)

plt.show()
#let's take a look at correlation matrix

def correlation_matrix(df):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cmap = cm.get_cmap('jet', 30)

    im = ax.matshow(df.corr().loc[:,('Time','Amount','Class')], interpolation="nearest", cmap=cmap)

    ax.grid(True)

    plt.title('PCA Correlation')

    

    #set major ticks but no labels

    ax.set_xticks(np.arange(-.5, 3, 1))

    ax.set_xticklabels('')

    #put labels in minor ticks

    ax.set_xticks(np.arange(0, 3, 1), minor=True)

    ax.set_xticklabels(['Time','Amount','Class'],fontsize=12, minor=True)

    

    #repeat for y-axis

    ax.set_yticks(np.arange(-.5, 31, 1))

    ax.set_yticklabels('')

    ax.set_yticks(np.arange(0, 31, 1), minor=True)

    ax.set_yticklabels(card_data.columns.values,fontsize=12, minor=True)

    

    ax.set_aspect('auto')

    

    #format the colorbar

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="1%", pad=0.05)

    fig.colorbar(im, cax=cax)

    plt.show()



correlation_matrix(card_data)