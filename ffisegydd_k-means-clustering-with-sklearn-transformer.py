# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.cluster import KMeans
class NeighbourhoodCluster(BaseEstimator, TransformerMixin):

    def __init__(self, num_clusters, seed):

        self.num_clusters = num_clusters

        self.seed = seed

        self.cluster = KMeans(self.num_clusters, random_state=self.seed)

        self.keys = ['latitude', 'longitude']



    def fit(self, X, y=None, **args):

        # Better method of removing outliers can be found in the notebook

        # https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/unsupervised-and-supervised-neighborhood-encoding

        mask = (

            (X["longitude"] < -73.75) & 

            (X["longitude"] > -74.05) & 

            (X["latitude"] > 40.4) & 

            (X["latitude"] < 40.9)

        )

        self.cluster.fit(X[mask][self.keys])

        return self



    def transform(self, X, y=None, **args):

        clusters = self.cluster.predict(X[self.keys])

        return clusters[:, None]



    def get_feature_names(self):

        return ['neighbourhood']
# Load in train and test data

train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')



train['train'] = 1

test['train'] = 0



X = pd.concat([train, test])

y = X.pop('interest_level')
num_neighbourhoods = 10

seed = 42

transformer = NeighbourhoodCluster(num_neighbourhoods, seed)
neighbourhoods = transformer.fit_transform(X)
neighbourhoods[:10]