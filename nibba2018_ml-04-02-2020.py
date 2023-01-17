import numpy as np

import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()

iris_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],

                     columns= iris['feature_names'] + ['target'])

iris_data['target'] = iris_data['target'].astype(int)

iris_data.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
Y = iris_data['target']

X = iris_data.drop(['target'], axis = 1)
X.head()
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
k_neighbours = KNeighborsClassifier(algorithm='auto')

k_neighbours.fit(X_train, Y_train)
k_neighbours.score(X_test, Y_test) * 100
class knnClassifier():

    def __init__(self, X_train, Y_train):

        self.train_x = X_train.copy(deep=True)

        self.train_y = Y_train.copy(deep=True)

        self.knn = 5

        

    def euclidean_distance(self, row1, row2):

        distance = np.sqrt(np.sum((row1-row2)**2))

        return distance

    

    def predict_row(self, test_row):

        distances = np.zeros(len(self.train_x))

        

        for i,train_row in enumerate(self.train_x.index):

            distances[i] = self.euclidean_distance(train_row, test_row)

        

        return np.argmin(distances)



    def predict(self, X_test):

        self.predictions = []

        for test_row in X_test.index:

            self.predictions.append(self.predict_row(X_test.loc[test_row]))

            

        return self.predictions
knnClass = knnClassifier(X_train, Y_train)

knnClass.predict(X_test)
new_flower = {'sepal length (cm)': 7.4, "sepal width (cm)": 3.3, "petal length (cm)": 4.8, "petal width (cm)": 2.3}

new_flower = pd.DataFrame(columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], data=[[7.4, 3.3, 4.8, 2.3]])

new_flower
knnClass.predict_row(new_flower.loc[0])
X_train
Y_train
lol = [1,2,3,-23,-56]
lol.sort()
lol