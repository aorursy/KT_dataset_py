from pathlib import Path



import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

import seaborn

%matplotlib inline
input_dir = Path('../input')



train = pd.read_csv(input_dir / 'train.csv', index_col=0)



# Split targets from inputs.

train_targets = train[['target']].copy()

train_inputs = train.drop('target', axis=1)



train.head()
test_inputs = pd.read_csv(input_dir / 'test.csv', index_col=0)

test_inputs.head()
# Check dimensions

train_inputs.shape, test_inputs.shape
fig, ax = plt.subplots(figsize=(15, 10))



# Plot individual histograms.

for col in train_inputs:

    x = train_inputs[col].values 

    freq, bins = np.histogram(x, bins=10)

    ax.plot(bins[:-1], freq, color='gray', alpha=0.1)

    

# Plot normal distribution for shape comparison.

def gaussian(x, m=0, s=1, norm=1):

    return norm * np.exp(-((x - m) ** 2) / s)



# Range for plotting.

minimum = train_inputs.values.min()

maximum = train_inputs.values.max()

x = np.linspace(minimum, maximum, 10)



# Handpicked values for illustration.

mean = -0.25

stdev = 2

norm = 55

y = gaussian(x, m=mean, s=stdev, norm=norm)

ax.plot(x, y, color='r', label='Gaussian(-0.25, 2)')



ax.set_xlabel('Value')

ax.set_ylabel('Freq')

ax.set_title('Distribution of training variables')

ax.legend(loc='upper right')

ax.grid()

from scipy.stats import shapiro



# Test for normality in each column using Shapiro-Wilks.

non_norm_cols = []

for col in train_inputs:

    x = train_inputs[col].values

    stat, pval = shapiro(x)

    

    if pval < 0.05:

        print('P-val {0:.2f} ==> evidence column {1} not normally distributed'.format(pval, col))

        non_norm_cols.append(col)

        
fig, ax = plt.subplots(figsize=(15, 8))

train_inputs[non_norm_cols].hist(ax=ax, bins=10);
fig, ax = plt.subplots(figsize=(12, 10))

seaborn.heatmap(train_inputs.corr(), cmap='bwr', vmin=-1, vmax=1, ax=ax)
from sklearn.decomposition import PCA



# Try different numbers of components and plot falloff in variance.

pca = PCA(n_components=200)

x = train_inputs.values

x_trans = pca.fit_transform(x)

n_comps = list(range(0, 200))

var = []

    

for n in n_comps:

    var.append(pca.explained_variance_ratio_[:n].sum())

    

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(n_comps, var)

ax.set_xlabel('N components')

ax.set_ylabel('Sum of explained variance')

ax.set_title('Explained variance versus number of principle components')

ax.grid()