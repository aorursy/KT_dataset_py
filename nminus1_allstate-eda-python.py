import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

dftrain = pd.read_csv('../input/train.csv')



# Get Categorical Columns

nonnumcols=[]

for col in dftrain.columns:

    if col.startswith('cat'):

        nonnumcols.append(col)



# Get Categorical Columns

numcols=[]

for col in dftrain.columns:

    if col.startswith('cont'):

        numcols.append(col)
ncol = 3

nrow = 2

for i in range(nrow):

    fig,axs = plt.subplots(nrows=1,ncols=ncol,sharey=True,figsize=(12, 8))

    cols = nonnumcols[i*ncol:ncol*(i+1)]

    for i in range(len(axs)):

        axs[i].set(yscale="log")

        sns.boxplot(x=cols[i], y="loss", data=dftrain, ax=axs[i])

        axs[i].set(xlabel=cols[i], ylabel='log(loss)')
for i in range(nrow):

    fig,axs = plt.subplots(nrows=1,ncols=ncol,sharey=True,figsize=(12, 8))

    cols = numcols[i*ncol:ncol*(i+1)]

    for i in range(len(axs)):

        sns.distplot(dftrain[cols[i]], ax=axs[i], hist=False)

        xlabel=cols[i]+" ( R squared = "+str(round(np.corrcoef(dftrain.loss, dftrain[cols[i]])[0, 1],2))+" ) "

        axs[i].set(xlabel=xlabel, ylabel='Density')
for i in range(nrow):

    fig,axs = plt.subplots(nrows=1,ncols=ncol,sharey=True,figsize=(12, 8))

    cols = numcols[i*ncol:ncol*(i+1)]

    for i in range(len(axs)):

        axs[i].set(yscale="log")

        sns.regplot(x=cols[i], y="loss", data=dftrain, ax=axs[i])

        xlabel=cols[i]+" ( R squared = "+str(round(np.corrcoef(dftrain.loss, dftrain[cols[i]])[0, 1],2))+" ) "

        axs[i].set(xlabel=xlabel, ylabel='log(loss)')
d = dftrain[numcols]

corrmat = d.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)