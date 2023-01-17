# Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings('ignore')



# Read raw data from the file



import pandas

import numpy as np

import random

import matplotlib.pylab as plt

from matplotlib.ticker import MaxNLocator

import pylab as p



train = pandas.read_csv("../input/train.csv")

plt.rcParams['figure.figsize'] = 10, 6 #[6.0, 4.0]
def plot_feature_loss(input_df,feature_name = 'cont1',num_bins = 50):

    train_temp = input_df.copy()

    feature_name_binned = feature_name + '_binned'

    bins = np.linspace(0,1.0,num_bins)

    train_temp[feature_name_binned] = np.digitize(train_temp[feature_name],bins=bins,right=True)

    train_temp[feature_name_binned] = train_temp[feature_name_binned] / num_bins

    cont_14_dict = train_temp.groupby(feature_name_binned)['loss'].mean().to_dict()

    cont_14_err_dict = train_temp.groupby(feature_name_binned)['loss'].sem().to_dict()

    lists = sorted(cont_14_dict.items())

    x, y = zip(*lists)

    lists_err = sorted(cont_14_err_dict.items())

    x_err, y_error = zip(*lists_err)



    p.figure()

    plt.errorbar(x,y,fmt = 'o',yerr = y_error,label = feature_name)

    p.xlabel(feature_name,fontsize=20)

    p.ylabel('loss (mean)',fontsize=20)

    plt.tick_params(axis='both', which='major', labelsize=15)

    p.legend(prop={'size':20},numpoints=1,loc=(0.05,0.8))

    p.xlim([train_temp[feature_name].min() - 0.02, train_temp[feature_name].max() + 0.02 ])

    plt.grid()

    ax = plt.gca()



    plt.tick_params(axis='both', which='major', labelsize=15)

    ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))

    ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    del train_temp



for name in train.columns:

    if name.startswith('cont'):

        plot_feature_loss(train,feature_name = name)