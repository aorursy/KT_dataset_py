import numpy as np
import matplotlib.pyplot as plt

def generateDataSet(total_size):
    np.random.seed(100)
    train_size = int(total_size*3/4)
    test_size = int(total_size/4)
    train_data = np.reshape(np.linspace(0, 100, train_size) + np.random.uniform(-5, 5, train_size), (train_size, 1))
    train_target = np.reshape(np.linspace(0, 100, train_size) + np.random.uniform(-5, 5, train_size), (train_size, 1))
    test_data = np.reshape(np.linspace(0, 100, test_size) + np.random.uniform(-5, 5, test_size), (test_size, 1))
    test_target = np.reshape(np.linspace(0, 100, test_size) + np.random.uniform(-5, 5, test_size), (test_size, 1))
    return (train_data, train_target, test_data, test_target)
import matplotlib.pyplot as plt

train_data, train_target, test_data, test_target = generateDataSet(100)
plt.scatter(train_data, train_target)
plt.xlabel('x')
plt.xlabel('y')
plt.title("Training Data")
plt.show()
from sklearn import metrics
class DirectMethod:
    
    def __init__(self):
        self.m = 0
        self.sigX = 0
        self.sigY = 0
        self.sigXY = 0
        self.sigX2 = 0
        self.w0 = 0
        self.w1 = 0

    def fit(self, train_data, train_target):
        assert train_data.shape[0] == train_target.shape[0]
        for i in range(len(train_data)):
            self.m += 1
            self.sigX += train_data[i][0]
            self.sigY += train_target[i][0]
            self.sigXY += (train_data[i][0] * train_target[i][0])
            self.sigX2 += train_data[i][0] ** 2
        self.w1 = ((self.m * self.sigXY) - (self.sigX * self.sigY)) / ((self.m * self.sigX2) - (self.sigX ** 2))
        self.w0 = ((self.sigY - self.w1 * self.sigX)) / self.m

    def predict(self, test_data):
        test_predicted = []
        for data in test_data:
            test_predicted.append(self.w0 + self.w1 * data[0])
        return np.reshape(test_predicted, (len(test_data), 1))
# 0. Initialize the model
directMethodModel = DirectMethod()

# 1. Teach the machine
directMethodModel.fit(train_data, train_target)

# 2. Predict the value
test_predicted = directMethodModel.predict(test_data)

# 3. Calculate the error
metrics.mean_absolute_error(test_target, test_predicted)
class IterativeMethod:
    
    def __init__(self):
        self.learning_rate = 0.0001
        self.w = []

    def fit(self, train_data, train_target):
        self.w = [0] * (train_data.shape[1] + 1)
        m = train_data.shape[0]
        for i in range(100):
            train_predicted = []
            for j in range(train_data.shape[0]):
                train_predicted.append(self.w[0])
                for k in range(train_data.shape[1]):
                    train_predicted[j] += self.w[k+1] * train_data[j][k]
            gradient = 0
            for j in range(train_data.shape[0]):
                gradient += (train_predicted[j] - train_target[j][0])
            self.w[0] -= self.learning_rate * gradient / m
            for j in range(1, train_data.shape[1]+1):
                gradient = 0
                for k in range(train_data.shape[0]):
                    gradient += (train_predicted[k] - train_target[k][0]) * train_data[k][j-1]
                self.w[j] -= self.learning_rate * gradient / m

    def predict(self, test_data):
        test_predicted = []
        for i in range(test_data.shape[0]):
            test_predicted.append(self.w[0])
            for j in range(test_data.shape[1]):
                test_predicted[i] += self.w[j+1] * test_data[i][j]
        return np.reshape(test_predicted, (test_data.shape[0], 1))
# 0. Initialize the model
iterativeMethodModel = IterativeMethod()

# 1. Teach the machine
iterativeMethodModel.fit(train_data, train_target)

# 2. Predict the value
test_predicted = iterativeMethodModel.predict(test_data)

# 3. Calculate the error
metrics.mean_absolute_error(test_target, test_predicted)
# 0. Initialize the model
from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()

# 1. Teach the machine
linearRegression.fit(train_data, train_target)

# 2. Predict the value
test_predicted = linearRegression.predict(test_data)

# 3. Calculate the error
metrics.mean_absolute_error(test_target, test_predicted)