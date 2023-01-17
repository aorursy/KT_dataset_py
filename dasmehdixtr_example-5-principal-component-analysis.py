from sklearn.datasets import load_iris

import pandas as pd
iris = load_iris()



data = iris.data

feature_names = iris.feature_names

y = iris.target



df = pd.DataFrame(data,columns = feature_names)

df["sinif"] = y



x = data
from sklearn.decomposition import PCA



pca = PCA(n_components=2,whiten=True) #whitten = normalize

pca.fit(x)



x_pca = pca.transform(x)



print("variance ratio: ",pca.explained_variance_ratio_)



print("sum: ",sum(pca.explained_variance_ratio_))



print(pca.components_)
df["p1"] = x_pca[:,0]



df["p2"] = x_pca[:,1]



color = ["red","green","blue"]

import matplotlib.pyplot as plt



for each in range(3):

     plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])



plt.title('Two Dimensional Reducted PCA ( Variance Ratio = {:.3f})'.format(100*sum(pca.explained_variance_ratio_)))

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.grid()

plt.legend()

plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=1,whiten=True) #whitten = normalize

pca.fit(x)



x_pca = pca.transform(x)



print("variance ratio: ",pca.explained_variance_ratio_)



print("sum: ",sum(pca.explained_variance_ratio_))



print(pca.components_)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x_pca,y)



y_head = lr.predict(x_pca)
import numpy as np

import matplotlib.pyplot as plt

plt.scatter(np.arange(0,50),x_pca[:50,:], label = iris.target_names[0])

plt.scatter(np.arange(50,100), x_pca[50:100,:], label = iris.target_names[1])

plt.scatter(np.arange(100,150), x_pca[100:150,:], label = iris.target_names[2])

plt.plot(np.arange(0,150),y_head, label = 'predicted classes')

plt.xlabel('index of data')

plt.ylabel('points of reducted and normalized data')

plt.title('PCA example with first component ( Variance Ratio = {:.3f})'.format(100*sum(pca.explained_variance_ratio_)))

plt.legend()

plt.grid()

plt.show()