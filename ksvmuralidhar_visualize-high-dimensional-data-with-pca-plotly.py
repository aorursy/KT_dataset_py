import numpy as np

import pandas as pd

import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

%matplotlib inline
#Importing iris input features

iris = pd.read_csv("../input/iris/Iris.csv")

iris.drop(columns="Id",inplace=True)

iris.columns = ["Sepal Length", "Sepal Width", "Petal Length","Petal Width","y"]

Input = iris.iloc[:,:-1]
norm_X = MinMaxScaler().fit_transform(Input)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(norm_X)
evr = np.cumsum(pca.explained_variance_ratio_)

print(evr)
plt.plot(range(1,len(evr)+1),evr)

plt.xticks(range(1,len(evr)+1))

plt.title("Explained variance ratio")

plt.ylabel("Explained variance ratio")

plt.xlabel("n_components")

plt.show()
X_pca=pd.DataFrame(X_pca)

X_pca.columns=["pc1","pc2","pc3"]

X_pca["y"]=iris["y"]
fig = px.scatter_3d(X_pca, x='pc1', y='pc2', z='pc3',color='y',title="Iris 3D")

fig.update_traces(marker_coloraxis=None)

fig.show()