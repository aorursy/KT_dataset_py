# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

from sklearn.pipeline import make_pipeline

print(__doc__)



# Code source: Tyler Lanigan <tylerlanigan@gmail.com>

#              Sebastian Raschka <mail@sebastianraschka.com>



# License: BSD 3 clause



RANDOM_STATE = 42

FIG_SIZE = (10, 7)





features, target = load_wine(return_X_y=True)



# Make a train/test split using 30% test size

X_train, X_test, y_train, y_test = train_test_split(features, target,

                                                    test_size=0.30,

                                                    random_state=RANDOM_STATE)



# Fit to data and predict using pipelined GNB and PCA.

unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())

unscaled_clf.fit(X_train, y_train)

pred_test = unscaled_clf.predict(X_test)



# Fit to data and predict using pipelined scaling, GNB and PCA.

std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())

std_clf.fit(X_train, y_train)

pred_test_std = std_clf.predict(X_test)



# Show prediction accuracies in scaled and unscaled data.

print('\nPrediction accuracy for the normal test dataset with PCA')

print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))



print('\nPrediction accuracy for the standardized test dataset with PCA')

print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))



# Extract PCA from pipeline

pca = unscaled_clf.named_steps['pca']

pca_std = std_clf.named_steps['pca']



# Show first principal components

print('\nPC 1 without scaling:\n', pca.components_[0])

print('\nPC 1 with scaling:\n', pca_std.components_[0])



# Scale and use PCA on X_train data for visualization.

scaler = std_clf.named_steps['standardscaler']

X_train_std = pca_std.transform(scaler.transform(X_train))



# visualize standardized vs. untouched dataset with PCA performed

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)





for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):

    ax1.scatter(X_train[y_train == l, 0], X_train[y_train == l, 1],

                color=c,

                label='class %s' % l,

                alpha=0.5,

                marker=m

                )



for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):

    ax2.scatter(X_train_std[y_train == l, 0], X_train_std[y_train == l, 1],

                color=c,

                label='class %s' % l,

                alpha=0.5,

                marker=m

                )



ax1.set_title('Training dataset after PCA')

ax2.set_title('Standardized training dataset after PCA')



for ax in (ax1, ax2):

    ax.set_xlabel('1st principal component')

    ax.set_ylabel('2nd principal component')

    ax.legend(loc='upper right')

    ax.grid()



plt.tight_layout()



plt.show()
# Perform the necessary imports

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

import pandas as pd



# Reading grains data

grains = pd.read_csv('../input/seeds-width-vs-length.csv')

# Assign the 0th column of grains: width

width = grains.iloc[:,0]



# Assign the 1st column of grains: length

length = grains.iloc[:,1]



# Scatter plot width vs length

plt.scatter(width, length)

plt.xlabel('Width')

plt.ylabel('Length')

plt.axis('equal')

plt.show()



# Calculate the Pearson correlation

correlation, pvalue = pearsonr(width,length)



# Display the correlation

print(correlation)

# Import PCA

from sklearn.decomposition import PCA



# Create PCA instance: model

model = PCA()



# Apply the fit_transform method of model to grains: pca_features

pca_features = model.fit_transform(grains)



# Assign 0th column of pca_features: xs

xs = pca_features[:,0]



# Assign 1st column of pca_features: ys

ys = pca_features[:,1]



# Scatter plot xs vs ys

plt.scatter(xs, ys)

plt.xlabel('xs')

plt.ylabel('ys')

plt.axis('equal')

plt.show()



# Calculate the Pearson correlation of xs and ys

correlation, pvalue = pearsonr(xs, ys)



# Display the correlation

print("Correlation : ",correlation)

# Make a scatter plot of the untransformed points

plt.scatter(grains.iloc[:,0], grains.iloc[:,1])



# Create a PCA instance: model

model = PCA()



# Fit model to points

model.fit(grains)



# Get the mean of the grain samples: mean

mean = model.mean_



# Get the first principal component: first_pc

first_pc = model.components_[0,:]



# Plot first_pc as an arrow, starting at mean

plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)



# Keep axes on same scale

plt.axis('equal')

plt.xlabel('First Principal Component 1')

plt.ylabel('First Principal Component 2')

plt.show()

# Perform the necessary imports

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt





# Reading grains data

grains = pd.read_csv('../input/seeds.csv')

samples = grains.iloc[:,1:]

print("Samples Info ",samples.info())

# Preprocessing : Changing type of last column from int to float

s1=samples.iloc[:,-1]

s1=s1.astype(float)

samples.iloc[:,-1] = s1

print("Samples Info Updated ",samples.info())

# Create scaler: scaler

scaler = StandardScaler()



# Create a PCA instance: pca

pca = PCA()



# Create pipeline: pipeline

pipeline = make_pipeline(scaler,pca)



# Fit the pipeline to 'samples'

pipeline.fit(samples)



# Plot the explained variances

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('Variance')

plt.xticks(features)

plt.show()

# Import PCA

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



# scaled samples

scaler = StandardScaler()

scaled_samples = scaler.fit_transform(samples)

print('Scaled Samples \n',scaled_samples[1:5,:])



# Create a PCA model with 2 components: pca

pca = PCA(n_components=2)



# Fit the PCA instance to the scaled samples

pca.fit(scaled_samples)



# Transform the scaled samples: pca_features

pca_features = pca.transform(scaled_samples)



# Print the shape of pca_features

print("PCA Features Shape \n",pca_features.shape)



print("PCA Features \n", pca_features[1:10,:])
