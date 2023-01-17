# load the modules
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
# load/read the data
iris_df = pd.read_csv('../input/Iris.csv')
iris_df.info()
iris_df.sample(5)
# Species distribution
# iris_df.groupby('Species').size()
iris_df["Species"].value_counts()
# simple visualization to show how the inputs compare against each other
import seaborn as sns # visualization
sns.pairplot(data=iris_df, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='Species', diag_kind="hist")
# Add hue='Species' to colorize
# try diag_kind="kde" which stands for kernel density estimatation
# Separate features from targets
iris_X = iris_df.iloc[:,1:5]
iris_Y = iris_df.iloc[:,-1].replace({"Iris-setosa":1, "Iris-virginica":2, "Iris-versicolor":3})
#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans

n_clusters_array = range(1, 11)
wcss = [] # within-cluster sum of squares

for n_clusters in n_clusters_array:
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', max_iter = 300)
    kmeans.fit(iris_X)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(n_clusters_array, wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# PCA with two components
pca = PCA(n_components=2)
pca.fit(iris_X)
transform = pca.transform(iris_X)

print(sum(pca.explained_variance_ratio_))

plt.scatter(transform[:,0],transform[:,1], s=20, c = iris_Y, cmap = "nipy_spectral", edgecolor = "None")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.4)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
errors = []

for n_neighbors in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    errors.append(np.count_nonzero(knn.predict(X_test) - Y_test))

print(errors)
from sklearn import svm

X = transform
y = iris_Y

svc = svm.SVC(kernel='linear').fit(X, y)

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()