# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from pylab import rcParams

rcParams['figure.figsize'] = 20, 30

import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 20, 30

train_csv = pd.read_csv("../input/photometric-redshift-estimation-2019/train.csv")

test_csv = pd.read_csv("../input/photometric-redshift-estimation-2019/test.csv")

mytry_csv = pd.read_csv("../input/predictedredshifts/redshift2.csv")

train = np.copy(train_csv.values)

test = np.copy(test_csv.values)

pred = np.copy(mytry_csv.values)
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
rcParams['figure.figsize'] = 13, 13

reds_hist = plt.hist(train[:,9], 50,histtype="step",alpha= 1.0)

plt.title("Az tanulóhalmazbeli és a teszthalmazból becsült vöröseltolódások")

pred_red_hist = plt.hist(pred[:,1], 50,histtype="step",alpha= 1.0)