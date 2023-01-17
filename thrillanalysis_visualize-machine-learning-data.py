import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns

import os

print(os.listdir("../input"))
data = pd.read_csv("../input/diabetes.csv")
data.head()
data.hist(figsize=(16, 14))
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(16, 14))

plt.show()
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(16,14))

plt.show()
column = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']

for col in column:

    sns.set()

    fig, ax = plt.subplots()

    sns.set(style="ticks")

    sns.violinplot(x='Outcome', y=col, data=data)  # column is chosen here

    sns.despine(offset=10, trim=True) 

    fig.set_size_inches(22,14)

    plt.savefig('{}.pdf'.format(col), bbox_inches='tight')
g = sns.PairGrid(data)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6);
correlations = data.corr()

correlations = data.corr()

# plot correlation matrix

fig = plt.figure(figsize=(16,14))

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(data.columns)

ax.set_yticklabels(data.columns)

plt.show()
scatter_matrix(data, figsize=(16,14))

plt.show()