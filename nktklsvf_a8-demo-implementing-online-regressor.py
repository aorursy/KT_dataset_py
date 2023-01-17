import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.base import BaseEstimator

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score

from sklearn.model_selection import train_test_split

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler
class SGDRegressor(BaseEstimator):

    # you code here

    def __init__(self, eta=0.001, n_epochs=3):

        self.eta = eta

        self.n_epochs = 3

        self.mse_ = []

        self.weights_ = []

        

    def fit(self, X, y):

        X = np.hstack([np.ones([X.shape[0], 1]), X])

        w = np.zeros(X.shape[1])

        for i in range(self.n_epochs):

            for j in range(X.shape[0]):              

                w += self.eta * (y[j] - np.sum(w * X[j])) * X[j]

                self.weights_.append(w.copy())

                self.mse_.append((np.square(y - np.sum(X * w, axis=1)).mean()))

        mse_argmin = np.argmin(self.mse_)

        self.mse = self.mse_[mse_argmin]

        self.w_ = self.weights_[mse_argmin]

        return self

                  

    def predict(self, X):

        X = np.hstack([np.ones([X.shape[0], 1]), X])

        return np.sum(X * self.w_, axis=1)
data_demo = pd.read_csv('../input/weights_heights.csv')
plt.scatter(data_demo['Weight'], data_demo['Height']);

plt.xlabel('Weight (lbs)')

plt.ylabel('Height (Inch)')

plt.grid();
X, y = data_demo['Weight'].values, data_demo['Height'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                     test_size=0.3,

                                                     random_state=17)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.reshape([-1, 1]))

X_valid_scaled = scaler.transform(X_valid.reshape([-1, 1]))
# you code here

sgd_regressor = SGDRegressor()

sgd_regressor.fit(X_train_scaled, y_train)
plt.plot(range(len(sgd_regressor.mse_)), sgd_regressor.mse_)
# you code here

sgd_regressor.mse, sgd_regressor.w_
weights_transposed = np.array(sgd_regressor.weights_).transpose()

plt.plot(range(len(sgd_regressor.weights_)), weights_transposed[0])
plt.plot(range(len(sgd_regressor.weights_)), weights_transposed[1])
# you code here

sgd_holdout_mse = mean_squared_error(y_valid, sgd_regressor.predict(X_valid_scaled))

sgd_holdout_mse
# you code here

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

linreg_holdout_mse = mean_squared_error(y_valid, lr.predict(X_valid_scaled))

linreg_holdout_mse
try:

    assert (sgd_holdout_mse - linreg_holdout_mse) < 1e-4

    print('Correct!')

except AssertionError:

    print("Something's not good.\n Linreg's holdout MSE: {}"

          "\n SGD's holdout MSE: {}".format(linreg_holdout_mse, 

                                            sgd_holdout_mse))