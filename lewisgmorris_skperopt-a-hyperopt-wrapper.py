#install skperopt

!pip install "skperopt==0.0.3"
#import 

import pandas as pd

import numpy as np

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier



import skperopt as sk
#generate classification data

data = make_classification(n_samples=1000, n_features=10, n_classes=2)

X = pd.DataFrame(data[0])

y = pd.DataFrame(data[1])
#init the classifier

kn = KNeighborsClassifier()

param = {"n_neighbors": [int(x) for x in np.linspace(1, 60, 30)],

         "leaf_size": [int(x) for x in np.linspace(1, 60, 30)],

         "p": [1, 2, 3, 4, 5, 10, 20],

         "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],

         "weights": ["uniform", "distance"]}

#search parameters

search = sk.HyperSearch(kn, X, y, params=param,verbose=1)

search.search()
#apply best parameters

kn.set_params(**search.best_params)