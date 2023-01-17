# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#  defining the common plot for the figures 

def common_plot(ax, title):

    ax.set_xlabel('feature 1', color='blue')

    ax.set_ylabel('feature 2', color='blue')

    ax.set_title(title, color='gray')

   

    

#simple classification 

# plot the data

import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.svm import SVC



# create 50 separable points

X, y = make_blobs(n_samples=50, centers=2,

                  random_state=0, cluster_std=0.60)



# fit the support vector classifier model

clf = SVC(kernel='linear')

clf.fit(X, y)



# create some new points to predict

X2, _ = make_blobs(n_samples=80, centers=2,

                   random_state=0, cluster_std=0.80)

X2 = X2[50:]



# predict the labels

y2 = clf.predict(X2)

fig, ax = plt.subplots(figsize=(8, 6))

point_style = dict(cmap='Paired', s=50)

ax.scatter(X[:, 0], X[:, 1], c=y, **point_style)



# format plot

common_plot(ax, 'Input Data')

ax.axis([-1, 4, -2, 7])