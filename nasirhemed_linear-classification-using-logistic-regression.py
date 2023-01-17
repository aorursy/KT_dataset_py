import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
data = load_iris()
X, y = data['data'], data['target']
names = data['target_names']
zipped = dict(zip([0, 1, 2], names))
print(zipped)
print("The shape of X is {} x {}".format(X.shape[0], X.shape[1]))
print("The shape of y is {} x 1".format(y.shape[0]))
from collections import Counter
Counter(y)
X_petals = X[:, 2:]

plt.plot(X_petals[y == 0][:, 0], X_petals[y == 0][:, 1], 'b.', label='setosa')
plt.plot(X_petals[y == 1][:, 0], X_petals[y == 1][:, 1], 'r.', label='versicolor')
plt.plot(X_petals[y == 2][:, 0], X_petals[y == 2][:, 1], 'g.', label='virginica')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()

plt.show()
X_sepals = X[:, :2]

plt.plot(X_sepals[y == 0][:, 0], X_sepals[y == 0][:, 1], 'b.', label='setosa')
plt.plot(X_sepals[y == 1][:, 0], X_sepals[y == 1][:, 1], 'r.', label='versicolor')
plt.plot(X_sepals[y == 2][:, 0], X_sepals[y == 2][:, 1], 'g.', label='virginica')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

plt.show()
## Option 1 (Manual)
np.random.seed(0)
indices = np.arange(X.shape[0]) # Generate an array of indices 0, ..., N
np.random.shuffle(indices) # Shuffle this array
# Shuffle the actual dataset and maintain the mapping between data and target
shuffled_X = X[indices] 
shuffled_y = y[indices]
# Split the data set 
X_train, X_test = np.split(shuffled_X, [int(0.8*X.shape[0])])
y_train, y_test = np.split(shuffled_y, [int(0.8*X.shape[0])])
# Optiona 2 (Using sklearn train_test_split)
# We'll stick with this one for the rest of the notebook
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

X_train_only_petals = X_train[:, 2:]

classifier = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=300)
%time classifier.fit(X_train_only_petals, y_train)
y_pred = classifier.predict(X_test[:, 2:])
np.count_nonzero(y_pred - y_test)  / y_pred.shape[0]
# Binary Classification
y_bin = y_train
y_bin[y_train != 0] = -1
y_bin[y_train == 0] = 1
y_bin[y_train == -1] = 0

y_bin_test = y_test
y_bin_test[y_test != 0] = -1
y_bin_test[y_test == 0] = 1
y_bin_test[y_test == -1] = 0
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
X_train_only_sepals = X_train[:, :2]

model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X_train_only_sepals, y_bin)
predictions = model.predict(X_test[:, :2])
(predictions == y_bin_test).mean()

model.theta
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');