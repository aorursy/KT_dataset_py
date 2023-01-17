import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams



import sklearn

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale,normalize, StandardScaler

from sklearn import datasets

from sklearn.metrics import confusion_matrix, classification_report
iris = datasets.load_iris()
# print(iris.DESCR)
#X = scale(iris.data)

X=iris.data

y = pd.DataFrame(iris.target)
clustering = KMeans(n_clusters=3)

clustering.fit(X)
clustering.labels_
target_predicted = np.choose(clustering.labels_,[2,0,1]).astype(np.int64)

#target_predicted