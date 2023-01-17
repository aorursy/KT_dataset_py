import numpy as np

import pandas as pd

from pandas.plotting import scatter_matrix

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_blobs
data = pd.read_csv("../input/iris/Iris.csv")
data.head()
features_with_null= [feature for feature in data.columns if data[feature].isnull().sum() >1]
data["Species"].unique()
data.shape
data=data.drop(["Id"],axis=1)
data.shape
data_setosa = data.loc[data["Species"]=="Iris-setosa"]

data_virginica = data.loc[data["Species"]=="Iris-virginica"]

data_versicolor = data.loc[data["Species"]=="Iris-versicolor"]
plt.plot(data_setosa["SepalLengthCm"],np.zeros_like(data_setosa["SepalLengthCm"]),"o")

plt.plot(data_virginica["SepalLengthCm"],np.zeros_like(data_virginica["SepalLengthCm"]),"o")

plt.plot(data_versicolor["SepalLengthCm"],np.zeros_like(data_versicolor["SepalLengthCm"]),"o")

plt.show()
data_setosa["SepalLengthCm"].min()
sns.FacetGrid(data,hue="Species",size=5).map(plt.scatter,"SepalLengthCm","PetalLengthCm").add_legend();
sns.pairplot(data,hue="Species",size=3)

plt.show()
data.describe()
data["Species"].unique()
data.groupby("Species").size()
data.hist(figsize=(10,10))

plt.show()
data
#scatter_matrix(data,figsize=(13,13))

#lt.show()
le= preprocessing.LabelEncoder()
data["Species"]= le.fit_transform(data["Species"])

data["Species"].unique()
data
array=data.values

array
X=array[:,0:4]

y=array[:,4]
X
y
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= 0.2)
model= KNeighborsClassifier()
model.fit(X_test,y_test)
accuracy= model.score(X_test,y_test)
accuracy