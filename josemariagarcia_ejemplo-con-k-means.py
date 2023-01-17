# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt # Para la representación gráfica



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



blobs_3, classes_3 = make_blobs(300,

                               centers = 3,

                               cluster_std = 0.5,

                               random_state = 1)

color_map = np.array(['b','g','r','c','m','y','k'])

kmeans = KMeans(n_clusters = 3,

               random_state = 1).fit(blobs_3)

classes = kmeans.predict(blobs_3)



plt.scatter(blobs_3.T[0],

           blobs_3.T[1],

           marker = '.',

           s = 250)

print(classes)

plt.scatter(kmeans.cluster_centers_[:,0],

           kmeans.cluster_centers_[:,1],

           marker = '*',

           s = 250,

           color = 'black')





plt.show()