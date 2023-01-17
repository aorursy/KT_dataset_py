# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from pylab import rcParams

rcParams['figure.figsize'] = 12, 12

import matplotlib.pyplot as plt


train_csv = pd.read_csv("../input/photometric-redshift-estimation-2019/train.csv")

test_csv = pd.read_csv("../input/photometric-redshift-estimation-2019/test.csv")

mytry_csv = pd.read_csv("../input/ontrainredshifts/redshifTest.csv")

names = list(train_csv)

train = np.copy(train_csv.values)

test = np.copy(test_csv.values)

pred = np.copy(mytry_csv.values)

names
def feat_distros(sets,reso = 50):

    hists = []

    for i in range(sets[0].shape[1]):

        plt.subplot(sets[0].shape[1],1,i+1)

        bins = np.linspace(min(sets[0][:,i].min(), sets[1][:,i].min()), max(sets[0][:,i].max(), sets[1][:,i].max()), reso )

        for sett in sets:

            hists.append(plt.hist(sett[:,i], histtype="step", bins=bins, alpha= 0.7))

    plt.show()

    return hists
hists = feat_distros([test[:,:-1], train[:,:-1]], 100)


reds_hist = plt.hist(train[:,9], 100, (0,0.8),histtype="step",alpha= 1.0)

plt.title("The actual redshifts of the training set and the results of my model on it")

pred_red_hist = plt.hist(pred[:,1], 100, (0,0.8),histtype="step",alpha= 1.0)

plt.legend(["real labels", "estimated"])

plt.xlabel("redshift", fontsize = 14)

plt.ylabel("count",  fontsize = 14)
miss = pred[:,1]-train[:,9]

hist = plt.hist(miss, 400)

plt.xlabel("miss", fontsize = 14)

plt.ylabel("count",  fontsize = 14)

miss_plot = plt.plot([0,0], [0,45000])

histsq = np.histogram((miss)**2, 4000)

print(miss.shape[0])

n = histsq[0].cumsum()

plt.xlabel("N best results", fontsize = 14)

plt.ylabel("RMSE",  fontsize = 14)

cumu_hist = plt.plot(n, np.sqrt((histsq[0]*histsq[1][:-1]).cumsum()/n) )

bad = miss**2 > 0.08

print(bad.sum())

bounds = None

for idn in range(0, 10):

    plt.xlabel( names[idn]+" parameter", fontsize = 14)

    plt.ylabel("count",  fontsize = 14)

    bh = plt.hist(train[bad,idn], 100, range=bounds,histtype="step", alpha= 0.7, density=True)

    ah = plt.hist(train[:,idn],  100, range=bounds,histtype="step", alpha= 0.7, density = True)

    plt.legend(["bad misses", "whole trainig set"])

    plt.show()