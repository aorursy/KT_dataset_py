from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.25)
print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from scipy import stats
import numpy as np

class kNNClassifier:

    def __init__(self, neighbors):
        self.train_data = []
        self.train_target = []
        self.neighbors = neighbors

    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target

    def predict(self, test_data):
        test_predicted = []
        for i in range(len(test_data)):
            distances = []
            for j in range(len(self.train_data)):
                distances.append((distance.euclidean(self.train_data[j], test_data[i]), self.train_target[j]))
            distances.sort(key = lambda x: x[0])
            test_predicted.append(stats.mode(np.array(distances[:self.neighbors]).transpose()[1]).mode[0])
        return test_predicted
neigh = kNNClassifier(5)
neigh.fit(train_data, train_target)
test_predicted = neigh.predict(test_data)
print(accuracy_score(test_target, test_predicted))
from sklearn.neighbors import KNeighborsClassifier
kNeighborsClassifier = KNeighborsClassifier()
kNeighborsClassifier.fit(train_data, train_target)
test_predicted = kNeighborsClassifier.predict(test_data)
print(accuracy_score(test_target, test_predicted))