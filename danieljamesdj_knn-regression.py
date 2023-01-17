from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.25)
print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)
from sklearn import metrics
from scipy.spatial import distance

class kNNRegressor:

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
            mean = 0
            for j in range(self.neighbors):
                mean += distances[j][1]
            mean /= self.neighbors
            test_predicted.append(mean)
        return test_predicted
neigh = kNNRegressor(5)
neigh.fit(train_data, train_target)
test_predicted = neigh.predict(test_data)
metrics.mean_absolute_error(test_target, test_predicted)
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(train_data, train_target)
test_predicted = neigh.predict(test_data)
metrics.mean_absolute_error(test_target, test_predicted)