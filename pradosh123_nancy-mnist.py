# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import module
import sklearn
import matplotlib.pyplot as plt
#import Mnist from keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#checking the shpe
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#create filter for only for 0 and 1
filter_train = np.where((y_train == 0 ) | (y_train == 1))
filter_test = np.where((y_test == 0) | (y_test == 1))
#apply filter for loadind data only 0 and 1 
x_train, y_train = x_train[filter_train], y_train[filter_train]
x_test, y_test = x_test[filter_test], y_test[filter_test]
#checking the shpe
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print(np.unique(y_train))
print(np.unique(y_test))

# converting image to 1 D array
x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)


#normalize the data
x_train = x_train.astype(float) / 255.0
print(x_train.shape)

# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)


y_kmeans = kmeans.fit_predict(x_train)
np.unique(y_kmeans)

from sklearn import metrics
def calculate_metrics(estimator, data, labels):
    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))
# yu can calclate homogenity and completeness as well to see how well the model performed
calculate_metrics(kmeans,x_train,y_train)
# Clusters are represnted and 0 and 1 and don't confuse it with the predictied  0 or 1 value
# to find which cluster beongs to which digit do a group by and check the count for each digit

check_cluster= pd.DataFrame(y_train,columns=['original'])
check_cluster['cluster']=y_kmeans
check_cluster.head()
check_cluster.groupby(['original','cluster'])['cluster'].count()
# from the abve we can see that cluster 0 is actually digit 1 and cluster 1 is actually digit 0
# you can automate this and calculate accuracy
