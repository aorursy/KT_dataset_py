# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets #builin data set

import sklearn.metrics as sm

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline
iris=datasets.load_iris()

x=scale(iris.data)

y=pd.DataFrame(iris.target)

variable_names= iris.feature_names

x[0:10,]
clustering=KMeans(n_clusters=3, random_state=5)

clustering.fit(x)
iris_df=pd.DataFrame(iris.data)

iris_df.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y.columns=['Target']
color_theme=np.array(['darkgrey','lightsalmon', 'powderblue'])

plt.subplot(1,2,1)

plt.scatter(x=iris_df['Petal_Length'], y=iris_df['Petal_Width'], c=color_theme[iris.target], s= 50)

plt.title('Ground Truth Classification')

plt.subplot(1,2,2)                     

plt.scatter(x=iris_df['Petal_Length'], y=iris_df['Petal_Width'], c=color_theme[clustering.labels_], s= 50)

plt.title('KMeans Classification')
relabel=np.choose(clustering.labels_,[2,0,1]).astype(np.int64)

plt.subplot(1,2,1)

plt.scatter(x=iris_df['Petal_Length'], y=iris_df['Petal_Width'], c=color_theme[iris.target], s= 50)

plt.title('Ground Truth Classification')

plt.subplot(1,2,2)                     

plt.scatter(x=iris_df['Petal_Length'], y=iris_df['Petal_Width'], c=color_theme[relabel], s= 50)

plt.title('KMeans Classification')
print(classification_report(y, relabel))
import scipy

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import cophenet

from scipy.spatial.distance import pdist

from sklearn.cluster import AgglomerativeClustering
np.set_printoptions(precision=4, suppress=True)
cars=pd.read_csv('../input/praactice-data/mt1cars.csv')

cars.head()
x=cars[['mpg','disp', 'hp', 'wt']].values

y=cars[['am']].values
z=linkage(x, 'ward')
dendrogram(z, truncate_mode='lastp',  p=12,  leaf_rotation = 45, leaf_font_size= 15, show_contracted = True)

plt.title('Truncated Hierarchical Clustering Dendrogram')

plt.xlabel('Cluster Size')

plt.ylabel('Distance')

plt.axhline(y=500)

plt.axhline(y=150)
k=2

Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')

Hclustering.fit(x)

sm.accuracy_score(y, Hclustering.labels_)
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')

Hclustering.fit(x)

sm.accuracy_score(y, Hclustering.labels_)
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average')

Hclustering.fit(x)

sm.accuracy_score(y, Hclustering.labels_)
Hclustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average')

Hclustering.fit(x)

sm.accuracy_score(y, Hclustering.labels_)
import urllib

from sklearn.neighbors import KNeighborsClassifier

from sklearn import neighbors

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
xx=preprocessing.scale(x)
x_train, x_test, y_train, y_test = train_test_split(xx, y , test_size = 0.33, random_state= 17)
clf=KNeighborsClassifier()

clf.fit(x_train, y_train)

print(clf)
y_pred=clf.predict(x_test)

print(classification_report(y_test, y_pred))