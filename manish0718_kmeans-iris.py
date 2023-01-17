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
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
iris=datasets.load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data1.head()
plt.scatter(iris.data[:,0],iris.data[:,1])
plt.grid(True)
plt.show()
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=3)
clf.fit(iris.data)
clf.labels_
clf.cluster_centers_
plt.scatter(iris.data[:,0],iris.data[:,1],c=clf.labels_)
plt.scatter(clf.cluster_centers_[:,0],clf.cluster_centers_[:,1],c='red')
plt.grid(True)
plt.show()
