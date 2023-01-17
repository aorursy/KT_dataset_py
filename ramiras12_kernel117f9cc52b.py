import numpy as np

from sklearn import datasets

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, explained_variance_score

from sklearn.utils import shuffle 
from sklearn.datasets import load_boston

data=load_boston()

X=data.data

y=data.target
num_training = int(0.8 * len(X))

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:] 
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)

y_test_pred = sv_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)

evs = explained_variance_score(y_test, y_test_pred) 

print("\n#### Performance ####")

print("Mean squared error =", round(mse, 2))

print("Explained variance score =", round(evs, 2))
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]

print("\nPredicted price:", sv_regressor.predict([test_data])[0])