import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/Iris.csv")
data.info()
data.columns
data.corr()
data.describe()
data.head()
data.Species.unique()
setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]
setosa.Species = [0 if i == "Iris-setosa" else 0 for i in setosa.Species]
versicolor.Species = [1 if i == "Iris-versicolor" else 1 for i in versicolor.Species]
virginica.Species = [2 if i == "Iris-virginica" else 2 for i in virginica.Species]
data = pd.concat([setosa, versicolor, virginica],axis=0)
plt.figure(figsize=(11,7))
plt.plot(data.PetalLengthCm[data.Species == 0], label="setosa")
plt.plot(data.PetalLengthCm[data.Species == 1], label="versicolor")
plt.plot(data.PetalLengthCm[data.Species == 2], label="virginica")
plt.legend()
plt.xlabel("Petal Length Cm")
plt.ylabel("Id")
plt.show()
plt.figure(figsize=(11,7))
plt.scatter(data.PetalLengthCm[data.Species == 0], data.PetalWidthCm[data.Species == 0], label="setosa")
plt.scatter(data.PetalLengthCm[data.Species == 1], data.PetalWidthCm[data.Species == 1], label="versicolor")
plt.scatter(data.PetalLengthCm[data.Species == 2], data.PetalWidthCm[data.Species == 2], label="virginica")
plt.legend()
plt.xlabel("Petal Length Cm")
plt.ylabel("Petal Width Cm")
plt.show()
x = versicolor.PetalLengthCm.values.reshape(-1,1)
y = versicolor.PetalWidthCm.values.reshape(-1,1)

linear_reg = LinearRegression()
linear_reg.fit(x,y)
y_head = linear_reg.predict(x)

print("score :",linear_reg.score(x,y))

plt.figure(figsize=(11,7))
plt.scatter(x,y)
plt.plot(x,y_head, color="orange")
plt.xlabel("Petal Length Cm")
plt.ylabel("Petal Width Cm")
plt.show()
new_data = data.iloc[:,3:5]
plt.figure(figsize=(11,7))
plt.scatter(new_data.PetalLengthCm,new_data.PetalWidthCm)
plt.xlabel("Petal Length Cm")
plt.ylabel("Petal Width Cm")
plt.show()
wcss = []
for k in range(1,21):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,21), wcss)
plt.show()
kmeans = KMeans(n_clusters=3)
cluster = kmeans.fit_predict(data)
data["sinif"] = cluster

data.sinif.value_counts()
x = data.iloc[:,1:5]
y = data.Species

pca = PCA(n_components=2)
pca.fit(x,y)
x_pca = pca.transform(x)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
data["p1"] = x_pca[:,0]
data["p2"] = x_pca[:,1]
color = ["blue","orange","green"]

for i in range(3):
    plt.scatter(data.p1[data.Species==i], data.p2[data.Species==i], color=color[i])
x = (x - np.min(x))/(np.max(x)-np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3)
knn = KNeighborsClassifier()
grid = {"n_neighbors":np.arange(1,51)}
knn_cv = GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x,y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)



