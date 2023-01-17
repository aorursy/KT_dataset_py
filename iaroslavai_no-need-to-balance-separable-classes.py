import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Heavily disballanced, but separable assignment of class labels
X = np.random.rand(100000, 2)
y = np.sum(np.abs(X), axis=-1) < 0.05
X = (X.T - 0.1*y.T).T # perfectly separable classes.

print('Positive instances: %s' % np.sum(y==True))
print('Negative instances: %s' % np.sum(y==False))
print('Positive to negative class ratio: %s' % (np.sum(y==True) / np.sum(y==False)))

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Linear SVC with default parameters
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print("Model test accuracy: %s (1.0 is perfectly accurate)" % score)

# visualize data and predictions
def plot_data(X, y, title):
    plt.title(title)
    plt.scatter(X[~y, 0], X[~y, 1])
    plt.scatter(X[y, 0], X[y, 1])

# Classes are classified perfectly!
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plot_data(X_train, y_train, 'Training data')
plt.subplot(1, 3, 2)
plot_data(X_test, y_test, 'Testing data')
plt.subplot(1, 3, 3)
plot_data(X_test, y_pred, 'Model predictions')
plt.show()