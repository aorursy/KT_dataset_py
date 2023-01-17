## load the iris data into a DataFrame

import pandas as pd







col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

## map each iris species to a number with a dictionary and list comprehension.

iris = pd.read_csv('../input/iris.csv', header=None, names=col_names)

iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

iris['species_num'] = [iris_class[i] for i in iris.species]



iris.head(10)

## map each iris species to a number with a dictionary and list comprehension.

iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

iris['species_num'] = [iris_class[i] for i in iris.species]

## Create an 'X' matrix by dropping the irrelevant columns.

X = iris.drop(['species', 'species_num'], axis=1)

y = iris.species_num
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train, y_train)

pred=knn.predict(X_test)

acc = accuracy_score(pred, y_test)

print(acc)