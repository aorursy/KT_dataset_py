# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

iris = pd.read_csv("../input/Iris.csv")
iris.head()
#iris.ix[:,1:5].head()
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(iris.ix[:,1:5])
print(kmeans.labels_)
print(kmeans.cluster_centers_)
iris["KMeans"] = kmeans.labels_
sns.FacetGrid(iris, hue="KMeans", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
neighbors = KNeighborsClassifier(n_neighbors=3)
neighbors.fit(iris.ix[:,1:5], iris.ix[:,6])
print(neighbors)

neighbors.predict(iris.ix[:,1:5])
#sns.FacetGrid(iris, hue="KNN", size=5) \
#   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
#   .add_legend()