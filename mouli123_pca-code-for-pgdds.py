import pandas as pd

import numpy as np
data=pd.read_csv("../input/train_PCA.csv")
data.head()
data.shape
features=data.drop('SalePrice',axis=1)

Y=data['SalePrice']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(data)

pcadata = scaler.transform(data)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

pca = PCA().fit(pcadata)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.show()
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=40)

Y_sklearn = sklearn_pca.fit_transform(pcadata)
Y_sklearn.shape
print(sklearn_pca.explained_variance_)
import sklearn.model_selection as ms

import sklearn.linear_model as lm
x_train,x_test,y_train,y_test=ms.train_test_split(Y_sklearn,Y,test_size=0.2,random_state=22)
glm=lm.LinearRegression()
glm.fit(x_train,y_train)
glm.score(x_train,y_train)
glm.score(x_test,y_test)