import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

iris = pd.read_csv("../input/Iris.csv") 

iris.head()

sns.pairplot(iris.drop(labels=['Id'], axis=1), hue='Species')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(iris[['SepalLengthCm', 'SepalWidthCm', 
                                                        'PetalLengthCm', 'PetalWidthCm']],
                                                    iris['Species'], random_state=0)
print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm=svm.predict(X_test)

print("Test set score KNN: {:.2f}".format(knn.score(X_test, y_test)))
iris_with_pred = pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted_KNN', index=X_test.index), pd.Series(y_pred_svm, name='Predicted_SVM', index=X_test.index)], 
          ignore_index=False, axis=1)
print("Test set score SVM: {:.2f}".format(svm.score(X_test, y_test)))
#compare results via diagrams
fig, ax = plt.subplots(1,3, figsize=(15,5))
sns.countplot(x="Species", data=iris_with_pred, ax=ax[0])
sns.countplot(x="Predicted_KNN", data=iris_with_pred, ax=ax[1])
sns.countplot(x="Predicted_SVM", data=iris_with_pred, ax=ax[2])
fig.show()
from sklearn.cluster import KMeans
iris_without_class = iris.iloc[:, [1, 2, 3, 4]].values
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(iris_without_class)
plt.figure(figsize=(8,8))
plt.scatter(iris_without_class[y_kmeans == 0, 0], iris_without_class[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(iris_without_class[y_kmeans == 1, 0], iris_without_class[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(iris_without_class[y_kmeans == 2, 0], iris_without_class[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend(prop={'size': 10})
array_of_species = iris['Species'].values
array_of_species_num = pd.factorize(array_of_species)[0]
plt.figure(figsize=(8,8))
plt.scatter(iris_without_class[array_of_species_num == 1,0], iris_without_class[array_of_species_num == 1,1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(iris_without_class[array_of_species_num == 0,0], iris_without_class[array_of_species_num == 0,1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(iris_without_class[array_of_species_num == 2,0], iris_without_class[array_of_species_num == 2,1], s = 100, c = 'green', label = 'Iris-virginica')

plt.legend(prop={'size': 10})
