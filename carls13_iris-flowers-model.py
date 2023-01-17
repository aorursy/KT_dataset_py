from sklearn.datasets import load_iris

iris = load_iris()
#X and y

X = iris.data

y = iris.target



feature_names = iris.feature_names

target_names = iris.target_names
#Split data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) #25% as test set
#Using K nearest neighbor algorithm

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
from sklearn import metrics

print(metrics.accuracy_score(y_test, y_predict))
#Using Decision tree

from sklearn.tree import DecisionTreeClassifier

knn2 = DecisionTreeClassifier()

knn2.fit(X_train, y_train)

y_predict2 = knn2.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict2))
#Test our model with a sample test

sample = [[3, 5, 4, 2], [2, 3, 5, 4]]

predictions = knn.predict(sample)

pred_species = [iris.target_names[p] for p in predictions]

print('My predictions are: ')

print(pred_species)
import joblib

joblib.dump(knn2, 'iris-brain.joblib')
model = joblib.load("iris-brain.joblib")

model.predict(X_test)



#Predict loaded model

sample = [[3, 5, 4, 2], [2, 3, 5, 4]]

predictions = model.predict(sample)

pred_species = [iris.target_names[p] for p in predictions]

print('My predictions are: ')

print(pred_species)
from sklearn.datasets import load_iris 

iris = load_iris() 

import matplotlib.pyplot as plt



# The indices of the features that we are plotting

x_index = 0

y_index = 1



# colorbar with the Iris target names

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])



#chart configurations

plt.figure(figsize=(5, 4))

plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)

plt.colorbar(ticks=[0, 1, 2], format=formatter)

plt.xlabel(iris.feature_names[x_index])

plt.ylabel(iris.feature_names[y_index])



plt.tight_layout()

plt.show()
features = iris.data.T



plt.scatter(features[2], features[3], alpha=0.2,

            s=100*features[3], c=iris.target, cmap='viridis') #https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

plt.xlabel(iris.feature_names[2])

plt.ylabel(iris.feature_names[3]);

plt.colorbar(ticks=[0, 1, 2], format=formatter)