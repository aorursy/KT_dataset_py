import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.base import BaseEstimator

from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score

from sklearn.model_selection import train_test_split

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler
class SGDRegressor(BaseEstimator):

    def __init__(self, eta=1e-3, n_epochs=3):

        self.eta=eta

        self.n_epochs=n_epochs

        self.mse_=[]

        self.weights_=[]

    def fit(self, X, y):

        X = np.hstack([np.ones([X.shape[0], 1]), X])

        w = np.zeros(X.shape[1])

        for it in tqdm(range(self.n_epochs)):

            for i in range(X.shape[0]):

                new_w = w.copy()

                new_w[0] += self.eta * (y[i] - w.dot(X[i, :]))

                for j in range(1, X.shape[1]):

                    new_w[j] += self.eta * (y[i] - w.dot(X[i, :])) * X[i, j]  

                w = new_w.copy()

                self.weights_.append(w)

                self.mse_.append(mean_squared_error(y, X.dot(w)))

        self.w_ = self.weights_[np.argmin(self.mse_)]

        return self   

    def predict(self, X):

        X = np.hstack([np.ones([X.shape[0], 1]), X])

        return X.dot(self.w_)               
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
sgd_r=SGDRegressor()

sgd_r.fit(X_train_scaled, y_train)
plt.plot(range(len(sgd_r.mse_)), sgd_r.mse_)

plt.xlabel('#updates')

plt.ylabel('MSE');
np.min(sgd_r.mse_), sgd_r.w_
plt.subplot(121)

plt.plot(range(len(sgd_r.weights_)), 

         [w[0] for w in sgd_r.weights_]);

plt.subplot(122)

plt.plot(range(len(sgd_r.weights_)), 

         [w[1] for w in sgd_r.weights_]);
sgd_holdout_mse = mean_squared_error(y_valid, 

                                        sgd_r.predict(X_valid_scaled))

sgd_holdout_mse
from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train_scaled, y_train)

print(lm.coef_, lm.intercept_)

linreg_holdout_mse = mean_squared_error(y_valid, 

                                        lm.predict(X_valid_scaled))

linreg_holdout_mse
try:

    assert (sgd_holdout_mse - linreg_holdout_mse) < 1e-4

    print('Correct!')

except AssertionError:

    print("Something's not good.\n Linreg's holdout MSE: {}"

          "\n SGD's holdout MSE: {}".format(linreg_holdout_mse, 

                                            sgd_holdout_mse))