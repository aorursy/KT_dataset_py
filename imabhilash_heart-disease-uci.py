# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/heart.csv')
dataset.describe(include='all')
X=dataset.iloc[:,:].values
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)

dataset['Heart Disease']=y_kmeans
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,4], s = 100, c = 'red', label = '0')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 4], s = 100, c = 'blue', label = '1')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 4], s = 100, c = 'green', label = '2')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 4], s = 100, c = 'cyan', label = '3')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 4], s = 100, c = 'magenta', label = '4')

plt.title('Chances of getting Heart Disease')

plt.xlabel('Age ')

plt.ylabel('Cholestrol')

plt.legend()

plt.show()
dataset
dataset.to_csv('heart.csv')