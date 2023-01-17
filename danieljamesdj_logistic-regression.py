from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split



X, y = load_breast_cancer(return_X_y=True)

train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.25)

print(train_data.shape)

print(test_data.shape)

print(train_target.shape)

print(test_target.shape)
from sklearn.metrics import accuracy_score
import numpy as np



class LogisticRegressionDirect:

    

    def __init__(self):

        self.learning_rate = 0.0001

        self.iterations = 1000

        self.threshold = 0.5

        self.w = []

    

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

        

    def fit(self, train_data, train_target):

        self.w = np.zeros(train_data.shape[1] + 1)

        m = train_data.shape[0]

        for i in range(self.iterations):

            train_data_with_x0 = np.append(np.ones((m, 1)), train_data, axis=1)

            z = np.dot(train_data_with_x0, self.w)

            gradient = np.dot(train_target - self.sigmoid(z), train_data_with_x0)

            self.w += self.learning_rate * gradient / m

                

    def predict(self, test_data):

        m = test_data.shape[0]

        test_data_with_x0 = np.append(np.ones((m, 1)), test_data, axis=1)

        z = np.dot(test_data_with_x0, self.w)

        test_predicted = self.sigmoid(z)

        return test_predicted >= self.threshold
# 0. Initialize the model

logisticRegression = LogisticRegressionDirect()



# 1. Teach the machine

logisticRegression.fit(train_data, train_target)



# 2. Predict the value

test_predicted = logisticRegression.predict(test_data)



# 3. Calculate the error

print(accuracy_score(test_target, test_predicted))
from sklearn.linear_model import LogisticRegression

kNeighborsClassifier = LogisticRegression(fit_intercept=False, max_iter=1000)

kNeighborsClassifier.fit(train_data, train_target)

test_predicted = kNeighborsClassifier.predict(test_data)

print(accuracy_score(test_target, test_predicted))