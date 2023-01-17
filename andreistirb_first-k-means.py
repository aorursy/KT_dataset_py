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
# Importing the dataset

dataset = pd.read_csv('../input/Iris.csv')

X = dataset.iloc[:, [1,2,3,4]].values
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = kmeans.fit_predict(X)
import matplotlib.pyplot as plt



#Visualising the clusters sepal-length vs. sepal-width

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 150, c = 'red', label = 'Iris-versicolor')

plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 150, c = 'blue', label = 'Iris-setosa')

plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 150, c = 'yellow', label = 'Iris-virginica')

plt.title('Iris clusters')

plt.xlabel('sepal-length')

plt.ylabel('sepal-width')

plt.legend()

plt.show()
