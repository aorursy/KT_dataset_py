import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix
class StandardScaler:

    def __init__(self, with_mean=True, with_std=True):

        self.with_mean = with_mean

        self.with_std = with_std

        self._mean = None

        self._std_dev = None

    

    def _mean_func(self, d): return sum(d)/len(d)



    def _std_dev_func(self, d):

        d_mean = self._mean_func(d)

        numerator = 0

        for e in d:

            numerator += (e-d_mean)**2

        return (numerator/(len(d)-1))**(1/2)

    

    def _fit(self, X): return (X-self._mean)/self._std_dev

    

    def fit_transform(self, X, y=None):

        self._mean = self._mean_func(X)

        self._std_dev = self._std_dev_func(X)

        return self._fit(X)

    

    def transform(self, X, y=None):

        return self._fit(X)

    

    def inverse_transform(self, X): 

        return (X*self._std_dev)+self._mean
class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iter=1000):

        self.b = None   # when all feature inputs are zero (w0)

        self.w = None   # weights of each feature (w1x1 + w2x2 + ... + wnxn)

        self.learning_rate = learning_rate

        self.max_iter = max_iter

    

    def _sigmoid(self, z): return 1/(1+np.exp(-z))



    def predict_proba(self, X):

        return self._sigmoid(self.b + np.dot(X, self.w))

    

    def predict(self, X):

        return np.array([1 if proba >= 0.5 else 0 for proba in self.predict_proba(X)])

    

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        self.b = 0

        self.w = np.zeros(n_features)



        for _ in range(self.max_iter):

            y_pred = self.predict_proba(X)



            D_b = (1/n_samples)*np.sum(y_pred-y)

            D_w = (1/n_samples)*np.dot(X.T, (y_pred-y))



            self.b -= D_b*self.learning_rate

            self.w -= D_w*self.learning_rate
dataset = load_breast_cancer()

X = dataset.data

y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_mat=cm, figsize=(2, 2))
sizes = np.array([cm[0][0]+cm[1][1], cm[0][1]+cm[1][0]])

labels = 'Correct', 'Incorrect'

colours = ['blue', 'red']



plt.pie(sizes, labels=labels, colors=colours, shadow=True)

plt.show()
print('Accuracy:', (cm[0][0]+cm[1][1])/len(y_pred))