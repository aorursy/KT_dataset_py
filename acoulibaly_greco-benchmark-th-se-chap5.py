# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

'''This module runs a 5-Fold CV for all the algorithms (default parameters) on

the movielens datasets, and reports average RMSE, MAE, and total computation

time.  It is used for making tables in the README.md file'''



from __future__ import (absolute_import, division, print_function,

                        unicode_literals)

import time

import datetime

import random



import numpy as np

import six

from tabulate import tabulate



import pandas as pd # by amc



from surprise import Dataset



from surprise import Reader

from surprise.model_selection import cross_validate

from surprise.model_selection import KFold

from surprise import NormalPredictor

from surprise import BaselineOnly

from surprise import KNNBasic

from surprise import KNNWithMeans

from surprise import KNNBaseline

from surprise import SVD

from surprise import SVDpp

from surprise import NMF

from surprise import SlopeOne

from surprise import CoClustering

#file_path = 'u.data'

#reader = Reader(line_format='user item rating timestamp', sep='\t')

#data = Dataset.load_from_file(file_path, reader=reader)



# The algorithms to cross-validate

classes = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,

           CoClustering, BaselineOnly, NormalPredictor)



# ugly dict to map algo names and datasets to their markdown links in the table

stable = 'http://surprise.readthedocs.io/en/stable/'

LINK = {'SVD': '[{}]({})'.format('SVD',

                                 stable +

                                 'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD'),

        'SVDpp': '[{}]({})'.format('SVD++',

                                   stable +

                                   'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp'),

        'NMF': '[{}]({})'.format('NMF',

                                 stable +

                                 'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF'),

        'SlopeOne': '[{}]({})'.format('Slope One',

                                      stable +

                                      'slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne'),

        'KNNBasic': '[{}]({})'.format('k-NN',

                                      stable +

                                      'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic'),

        'KNNWithMeans': '[{}]({})'.format('Centered k-NN',

                                          stable +

                                          'knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans'),

        'KNNBaseline': '[{}]({})'.format('k-NN Baseline',

                                         stable +

                                         'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline'),

        'CoClustering': '[{}]({})'.format('Co-Clustering',

                                          stable +

                                          'co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering'),

        'BaselineOnly': '[{}]({})'.format('Baseline',

                                          stable +

                                          'basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly'),

        'NormalPredictor': '[{}]({})'.format('Random',

                                             stable +

                                             'basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor'),

       #'ml-100k': '[{}]({})'.format('Movielens 100k',

        #                             'http://grouplens.org/datasets/movielens/100k'),

        #'ml-1m': '[{}]({})'.format('Movielens 1M',

               #                    'http://grouplens.org/datasets/movielens/1m'),

        }



# added by amc 



# Creation of the dataframe. Column names are irrelevant.

# voting procedure rating  

ratings_dict = {'itemID': [2, 5, 1, 3, 1, 1, 1, 4, 4, 2, 1, 5, 5, 4, 4, 5, 5, 1, 4, 1, 5, 1, 2, 5, 5, 4, 4, 4, 3, 3, 5, 4, 5, 5, 5, 3, 4, 2, 5, 5, 5, 5, 1, 5, 4, 3, 2, 2, 2, 5, 4, 1, 4, 4, 1, 2, 1, 2, 1, 2, 2, 4, 5, 5, 4, 2, 1, 3, 1, 4, 4, 1, 1, 4, 1, 2, 4, 1, 2, 2, 4, 3, 4, 4, 5, 2, 3, 3, 2, 3, 3, 3, 4, 4, 5, 2, 4, 4, 5, 2],

                'userID': [45, 32, 9, 45, 23, 9, 23, 23, 9, 2, 23, 32, 9, 23, 32, 45, 32, 32, 32, 32, 45, 23, 9, 9, 45, 23, 9, 2, 2, 23, 2, 2, 45, 9, 45, 32, 23, 2, 45, 32, 9, 32, 23, 23, 45, 32, 2, 9, 9, 23, 45, 45, 23, 45, 32, 23, 2, 9, 45, 32, 45, 23, 23, 45, 23, 23, 9, 32, 9, 23, 32, 2, 2, 32, 23, 45, 23, 9, 9, 32, 45, 9, 23, 45, 32, 32, 9, 23, 9, 45, 23, 32, 45, 2, 32, 2, 2, 2, 23, 23],

                'rating': [3, 4, 3, 3, 5, 4, 3, 5, 5, 3, 4, 4, 5, 5, 4, 3, 3, 5, 5, 4, 4, 3, 5, 4, 3, 3, 4, 4, 5, 3, 5, 3, 4, 5, 4, 3, 3, 4, 5, 3, 3, 4, 4, 5, 4, 5, 5, 3, 4, 3, 5, 3, 5, 4, 4, 3, 3, 5, 5, 5, 5, 4, 3, 5, 5, 5, 4, 3, 4, 4, 4, 3, 3, 4, 3, 4, 4, 4, 3, 3, 3, 5, 5, 4, 5, 3, 5, 3, 5, 4, 5, 4, 3, 4, 4, 3, 5, 5, 5, 5]}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# set RNG

np.random.seed(0)

random.seed(0)



#dataset = 'ml-100k'

#data = Dataset.load_builtin(dataset)

dataset = data # by amc 

kf = KFold(random_state=0)  # folds will be the same for all algorithms.



table = []

for klass in classes:

    start = time.time()

    out = cross_validate(klass(), data, ['rmse', 'mae'], kf)

    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))

    link = LINK[klass.__name__]

    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))

    mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))



    new_line = [link, mean_rmse, mean_mae, cv_time]

    print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf

    table.append(new_line)



header = ['LINK[dataset]',

          'RMSE',

          'MAE',

          'Time'

          ]

print(tabulate(table, header, tablefmt="pipe"))



# Any results you write to the current directory are saved as output.