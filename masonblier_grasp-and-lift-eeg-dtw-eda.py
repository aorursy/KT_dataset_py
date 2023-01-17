# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.spatial.distance import euclidean

from sklearn.base import BaseEstimator,ClassifierMixin

from sklearn.model_selection import train_test_split

import fastdtw # linear-complexity dtw

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# from https://github.com/llvll/motionml/blob/master/ip%5By%5D/motionml.ipynb

class KnnDtwClassifier(BaseEstimator, ClassifierMixin):

    """Custom classifier implementation for Scikit-Learn using Dynamic Time Warping (DTW)

       and KNN (K-Nearest Neighbors) algorithms.

       This classifier can be used for labeling the varying-length sequences, like time series

       or motion data.

       FastDTW library is used for faster DTW calculations - linear instead of quadratic complexity.

    """

    def __init__(self, n_neighbors=1):

        self.n_neighbors = n_neighbors

        self.features = []

        self.labels = []



    def get_distance(self, x, y):

        return fastdtw(x, y)[0]



    def fit(self, X, y=None):

        for index, l in enumerate(y):

            self.features.append(X[index])

            self.labels.append(l)

        return self



    def predict(self, X):

        dist = np.array([self.get_distance(X, seq) for seq in self.features])

        indices = dist.argsort()[:self.n_neighbors]

        return np.array(self.labels)[indices]



    def predict_ext(self, X):

        dist = np.array([self.get_distance(X, seq) for seq in self.features])

        indices = dist.argsort()[:self.n_neighbors]

        return (dist[indices],

                indices)
def loadDataSets(data_dir):

    datasets = {}

    for d in os.listdir(data_dir)[0:3]:

        name_parts = d.split("_")

        short_name = f"{name_parts[0]}_{name_parts[1]}"

        if not short_name in datasets:

            datasets[short_name] = {

                'events': pd.read_csv(os.path.join(data_dir, f"{short_name}_events.csv")),

                'data': pd.read_csv(os.path.join(data_dir, f"{short_name}_data.csv")),

            }

    return datasets

datasets = loadDataSets("../input/train/train")
# data = X

first_key = list(datasets.keys())[0]

print("rows: ", len(datasets[first_key]['data']))

datasets[first_key]['data'].head()
# events table = Y

print("rows: ", len(datasets[first_key]['events']))

datasets[first_key]['events'].head()
# X_all from data, drop id

# sets = [datasets[d]['data'].drop(['id'],axis=1) for d in datasets.keys()]

lens = [len(datasets[d]['data']) for d in datasets.keys()]

print(lens)

# X_all = np.stack(sets)

# print(X_all.shape)

# X_all.head()