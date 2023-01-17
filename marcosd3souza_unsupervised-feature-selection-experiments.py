# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

from sklearn.metrics.cluster import v_measure_score

from sklearn import preprocessing
# variance filter method

def apply_variance_filter(dataset, variance_threshold):

    if variance_threshold != 1:

        cut_point = round(variance_threshold * dataset.shape[1])



        variances = dataset.var()

        data = pd.DataFrame({"variance": variances, "feature": dataset.columns.values})

        data_sorted = data.sort_values(by="variance", ascending=False)



        data_filtered = data_sorted.iloc[0:int(cut_point), :]

        dataset = dataset.loc[:, data_filtered.loc[:, "feature"].values]



    return dataset
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# list all files on input directory

# import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

dataset = pd.read_csv("../input/SMK_CAN_dataset.csv", index_col=False, sep=" ");



y_true = list(dataset.loc[:, 'Y'])

dataset = dataset.drop(['Y'], axis=1)



dataset = pd.DataFrame(preprocessing.scale(dataset))
print('dataset before variance filter: ', dataset.shape[:])



k_means = KMeans(n_clusters=2, init='k-means++', n_init=50, max_iter=3000,

                     tol=0.0001, precompute_distances=True, verbose=0,

                     random_state=None, copy_x=True, n_jobs=-1)



k_means.fit(dataset)

y_predict = k_means.labels_



print('nmi: ', v_measure_score(y_true, y_predict))
# after variance filter

dataset = apply_variance_filter(dataset, 0.25)



print('dataset after variance filter: ', dataset.shape[:])



k_means = KMeans(n_clusters=2, init='k-means++', n_init=50, max_iter=3000,

                     tol=0.0001, precompute_distances=True, verbose=0,

                     random_state=None, copy_x=True, n_jobs=-1)



k_means.fit(dataset)

y_predict = k_means.labels_



print('nmi: ', v_measure_score(y_true, y_predict))