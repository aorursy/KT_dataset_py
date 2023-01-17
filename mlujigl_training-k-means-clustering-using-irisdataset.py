import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

Iris = pd.read_csv("../input/Iris.csv")
Iris.shape  #150 rows and 6 columns
Iris.columns
Iris.describe()
Iris_data = Iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Iris_target = Iris['Species']

Iris_target.value_counts()
Iris_target = Iris_target.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
Iris_target.value_counts()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
colors = np.array(['red', 'green', 'blue'])

plt.subplot(2, 2, 1)
plt.scatter(Iris_data['SepalLengthCm'], Iris_data['SepalWidthCm'],
            c=colors[Iris_target], s=40, 
           )
plt.title('Sepal Length vs Sepal Width')
labels = [' red:Iris-setosan\n green:Iris-versicolor\n blue: Iris-virginica']
plt.legend(labels, markerscale=0,ncol=1, loc='best', 
           bbox_to_anchor=[3., 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           )

plt.subplot(2,2,2)
plt.scatter(Iris_data['PetalLengthCm'], Iris_data['PetalWidthCm'], 
            c= colors[Iris_target], s=40)
plt.title('Petal Length vs Petal Width')

plt.subplot(2,2,3)
plt.scatter(Iris_data['SepalLengthCm'], Iris_data['PetalWidthCm'],
            c= colors[Iris_target], s=40)
plt.title('Sepal Length vs Petal Width')

plt.subplot(2,2,4)
plt.scatter(Iris_data['SepalLengthCm'], Iris_data['PetalLengthCm'], 
            c= colors[Iris_target], s=40)
plt.title('Sepal Length vs Petal Length')
X = Iris_data[['PetalLengthCm', 'PetalWidthCm']]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
colors = np.array(['red', 'green', 'blue'])

plt.scatter(X['PetalLengthCm'], X['PetalWidthCm'], c=colors[Iris_target], s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.scatter(centers[:,0],centers[:,1], marker='o', s = 500, linewidths=2, c='none')
plt.scatter(centers[:,0],centers[:,1], marker='x', s = 500, linewidths=2)
