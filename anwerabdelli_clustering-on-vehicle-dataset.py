import numpy as np 

import pandas as pd

from scipy import ndimage 

from scipy.cluster import hierarchy 

from scipy.spatial import distance_matrix 

from matplotlib import pyplot as plt 

from sklearn import manifold, datasets 

from sklearn.cluster import AgglomerativeClustering 

from sklearn.datasets.samples_generator import make_blobs 

%matplotlib inline
df = pd.read_csv("../input/cars_clus.csv")

print ("Shape of dataset: ", df.shape)

df.head(5)
df.dtypes
df[[ 'sales', 'resale', 'type', 'price', 'engine_s',

       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',

       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',

       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',

       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

df=df.dropna()

df = df.reset_index(drop=True)

print ("Shape of dataset after cleaning: ", df.size)

df.head(5)
featureset = df[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
from sklearn.preprocessing import MinMaxScaler

x = featureset.values #returns a numpy array

min_max_scaler = MinMaxScaler()

feature_mtx = min_max_scaler.fit_transform(x)

feature_mtx [0:5]