# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/wholesale-customers-data/Wholesale customers data.csv")
data.columns
data.drop(['Channel', 'Region'], axis = 1, inplace = True)
display(data.describe())
data.head(10)
data.shape
## Feature relevance



new_data = data.drop('Grocery', axis = 1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(new_data, data.Grocery, test_size = 0.25, random_state = 20)



from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor()

regressor = model.fit(X_train, y_train)

prediction = regressor.predict(X_test)



from sklearn.metrics import r2_score

score = r2_score(y_test, prediction)

print("the score is", score)
data.isnull().any()
sns.pairplot(data)
import seaborn as sns



sns.heatmap(data.corr(), annot = True)
np
sns.scatterplot(x = data['Fresh'], y = data['Frozen'])
sns.scatterplot(x = data['Milk'], y = data['Detergents_Paper'])
sns.countplot(data['Fresh'])
data_sub = data.iloc[:,2:8]

data_sub.head(5)
data_sub.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(data_sub)

data_sub = scaler.transform(data_sub)
data_sub = pd.DataFrame(data_sub, columns=data_sub.columns)
from sklearn.decomposition import PCA

pca_data = PCA(n_components = 6)

principalComponenets_data = pca_data.fit_transform(data)

principalComponenets_data
pca_data.components_
pca_data.explained_variance_ratio_
np.cumsum(pca_data.explained_variance_ratio_)
sns.lineplot(x = ['1', '2', '3', '4', '5', '6'], y = np.cumsum(pca_data.explained_variance_ratio_))



plt.xlabel('Principal components')

plt.ylabel('Explained Variance ratio')
x = data.iloc[:,2:8]

x
from sklearn.cluster import KMeans



wcss = []



for i in range(1,11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    

    km.fit(x)

    

    wcss.append(km.inertia_)

    

plt.plot(range(1,11), wcss)



plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)



y_means = km.fit_predict(x)
plt.scatter(x.iloc[:,3], x.iloc[:,4])



plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s = 300, c = 'red')
principal_Df = pd.DataFrame(data = principalComponenets_data, 

                            columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6'])
wcss = []



for i in range(1,11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    

    km.fit(principal_Df)

    

    wcss.append(km.inertia_)

    

plt.plot(range(1,11), wcss)



plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)



y_means = km.fit_predict(principal_Df)
plt.scatter(principal_Df.iloc[:,0], principal_Df.iloc[:,1])



plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s = 300, c = 'red')
from sklearn.preprocessing import normalize

data_scaled = normalize(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

data_scaled.head()
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

plt.axhline(y=6, color='r', linestyle='--')
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters = 2, affinity = "euclidean", linkage = "ward")

cluster.fit_predict(data_scaled)
plt.figure(figsize = (10, 7))

plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c = cluster.labels_)