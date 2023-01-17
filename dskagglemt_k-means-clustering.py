import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
data = pd.read_csv ('../input/wholesale-customers-data-set/Wholesale customers data.csv')

data.head()
data.info()
cat_f = ['Channel', 'Region']

cont_f = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
data.describe()

# data[cont_f].describe()
data = pd.get_dummies(data=data, columns=cat_f, drop_first = True)

data.head()
all_colmns = data.columns

all_colmns
scale = MinMaxScaler()

scale.fit(data)

data_scaled = scale.transform(data)

data_scaled = pd.DataFrame(data_scaled, columns = all_colmns)

data_scaled.head()
Sum_of_squared_distances = []



K = range(1,15)



for k in K:

    km = KMeans(n_clusters = k)

    km = km.fit(data_scaled)

    Sum_of_squared_distances.append(km.inertia_)

    
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



import sklearn

from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import scale



import sklearn.metrics as sm

from sklearn import datasets

from sklearn.metrics import confusion_matrix, classification_report
iris = datasets.load_iris()



X = scale(iris.data)



y = pd.DataFrame(iris.target)



variable_names = iris.feature_names



X[0:10,]
clustering = KMeans(n_clusters = 3, random_state = 5)



clustering.fit(X)
iris_df = pd.DataFrame(iris.data)



iris_df.columns = ['Speal_length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']



y.columns = ['Targets']
color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])



plt.subplot(1,2,1)



plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s = 50)



plt.title('Ground Truth CLassification')



plt.subplot(1,2,2)



plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[clustering.labels_], s = 50)



plt.title('K-Means Classification')

relabel = np.choose(clustering.labels_, [2,0,1]).astype(np.int64)

plt.subplot(1,2,1)



plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[iris.target], s = 50)



plt.title('Ground Truth CLassification')



plt.subplot(1,2,2)



plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c = color_theme[relabel], s = 50)



plt.title('K-Means Classification')