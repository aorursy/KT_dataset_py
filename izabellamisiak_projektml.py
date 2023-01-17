# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fish-market/Fish.csv')

df = data.copy()

df.head()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid', context='notebook')

sns.pairplot(df, height=2.5);

plt.show()
correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)

class LinearRegressionGD(object):

 def __init__(self, eta=0.001, n_iter=20):

    self.eta = eta

    self.n_iter = n_iter

 def fit(self, X, y):

    self.w_ = np.zeros(1 + X.shape[1])

    self.cost_ = []

    for i in range(self.n_iter):

      output = self.net_input(X)

      errors = (y - output)

      self.w_[1:] += self.eta * X.T.dot(errors)

      self.w_[0] += self.eta * errors.sum()

      cost = (errors**2).sum() / 2.0

      self.cost_.append(cost)

    return self

 def net_input(self, X):

    return np.dot(X, self.w_[1:]) + self.w_[0]

 def predict(self, X):

    return self.net_input(X)
X = df[['Length1']].values

y = df['Weight'].values

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)

y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()

lr.fit(X_std, y_std)
plt.plot(range(1, lr.n_iter+1), lr.cost_)

plt.ylabel('SSE')

plt.xlabel('Epoch')

plt.tight_layout()

plt.show()
def lin_regplot(X, y, model):

    plt.scatter(X, y, c='blue')

    plt.plot(X, model.predict(X), color='red', linewidth=2)    

    return
lin_regplot(X_std, y_std, lr)

plt.xlabel('Length1 (standardized)')

plt.ylabel('Weight (standardized)')

plt.tight_layout()

plt.show()



print('Slope: %.3f' % lr.w_[1])

print('Intercept: %.3f' % lr.w_[0])
length_std = sc_x.transform(np.array([[24.0]]))

weight_std = lr.predict(length_std)

print("Weight in gram: %.3f" % sc_y.inverse_transform(weight_std))
from sklearn.linear_model import LinearRegression

slr = LinearRegression()

slr.fit(X, y)

y_pred = slr.predict(X)

print('Slope: %.3f' % slr.coef_[0])

print('Intercept: %.3f' % slr.intercept_)



lin_regplot(X, y, slr)

plt.xlabel('Weight in grams')

plt.ylabel('Length1')

plt.tight_layout()

plt.show()
from sklearn.linear_model import RANSACRegressor



ransac = RANSACRegressor(LinearRegression(), 

                             max_trials=100, 

                             min_samples=50, 

                             residual_threshold=150.0, 

                             random_state=0)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_

outlier_mask = np.logical_not(inlier_mask)



line_X = np.arange(5, 60, 1)

line_y_ransac = ransac.predict(line_X[:, np.newaxis])

plt.scatter(X[inlier_mask], y[inlier_mask],

            c='blue', marker='o', label='Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask],

            c='lightgreen', marker='s', label='Outliers')

plt.plot(line_X, line_y_ransac, color='red')   

plt.xlabel('Weight in grams')

plt.ylabel('Length')

plt.legend(loc='upper left')



plt.tight_layout()

plt.show()



print('Slope: %.3f' % ransac.estimator_.coef_[0])

print('Intercept: %.3f' % ransac.estimator_.intercept_)

from sklearn.model_selection import train_test_split



X = df[['Length1','Length2','Length3','Height','Width']].values

y = df['Weight'].values



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=0)

slr = LinearRegression()



slr.fit(X_train, y_train)

y_train_pred = slr.predict(X_train)

y_test_pred = slr.predict(X_test)
plt.scatter(y_train_pred,  y_train_pred - y_train,

            c='blue', marker='o', label='Training data')

plt.scatter(y_test_pred,  y_test_pred - y_test,

            c='lightgreen', marker='s', label='Test data')

plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc='upper left')

plt.hlines(y=0, xmin=-10, xmax=1300, lw=2, color='red')

plt.xlim([-10, 1200])

plt.tight_layout()

plt.show()
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



print('MSE train: %.3f, test: %.3f' % (

        mean_squared_error(y_train, y_train_pred),

        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (

        r2_score(y_train, y_train_pred),

        r2_score(y_test, y_test_pred)))
from sklearn.preprocessing import PolynomialFeatures
X = df[['Length1']].values

y = df['Weight'].values



regr = LinearRegression()



# create quadratic features

quadratic = PolynomialFeatures(degree=2)

cubic = PolynomialFeatures(degree=3)

X_quad = quadratic.fit_transform(X)

X_cubic = cubic.fit_transform(X)



# fit features

X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]



regr = regr.fit(X, y)

y_lin_fit = regr.predict(X_fit)

linear_r2 = r2_score(y, regr.predict(X))



regr = regr.fit(X_quad, y)

y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))

quadratic_r2 = r2_score(y, regr.predict(X_quad))



regr = regr.fit(X_cubic, y)

y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))

cubic_r2 = r2_score(y, regr.predict(X_cubic))





# plot results

plt.scatter(X, y, label='training points', color='skyblue')



plt.plot(X_fit, y_lin_fit, 

         label='linear (d=1), $R^2=%.2f$' % linear_r2, 

         color='red', 

         lw=2, 

         linestyle=':')



plt.plot(X_fit, y_quad_fit, 

         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,

         color='green', 

         lw=2,

         linestyle=':')



plt.plot(X_fit, y_cubic_fit, 

         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,

         color='blue', 

         lw=2, 

         linestyle='-')



plt.xlabel('Length in cm')

plt.ylabel('Weight in grams')

plt.legend(loc='upper left')



plt.tight_layout()

plt.show()
X = df[['Length1']].values

y = df['Weight'].values



# transform features

X_log = np.log(X)

y_sqrt = np.sqrt(y)



# fit features

X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]



regr = regr.fit(X_log, y_sqrt)

y_lin_fit = regr.predict(X_fit)

linear_r2 = r2_score(y_sqrt, regr.predict(X_log))



# plot results

plt.scatter(X_log, y_sqrt, label='training points', color='skyblue')



plt.plot(X_fit, y_lin_fit, 

         label='linear (d=1), $R^2=%.2f$' % linear_r2, 

         color='blue', 

         lw=2)



plt.xlabel('Length in cm')

plt.ylabel('Weight in grams')

plt.legend(loc='upper left')



plt.tight_layout()

plt.show()



print('MSE train: %.3f, test: %.3f' % (

        mean_squared_error(y_train, y_train_pred),

        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (

        r2_score(y_train, y_train_pred),

        r2_score(y_test, y_test_pred)))
from sklearn.tree import DecisionTreeRegressor



X = df[['Length1']].values

y = df['Weight'].values



tree = DecisionTreeRegressor(max_depth=3)

tree.fit(X, y)



sort_idx = X.flatten().argsort()



lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('Length in cm')

plt.ylabel('Weight in grams')

plt.show()
X = df[['Length1','Length2','Length3','Height','Width']].values

y = df['Weight'].values



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=1)



from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(n_estimators=100, 

                               criterion='mse', 

                               random_state=1, 

                               n_jobs=-1)

forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)

y_test_pred = forest.predict(X_test)



print('MSE train: %.3f, test: %.3f' % (

        mean_squared_error(y_train, y_train_pred),

        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (

        r2_score(y_train, y_train_pred),

        r2_score(y_test, y_test_pred)))
plt.scatter(y_train_pred,  

            y_train_pred - y_train, 

            c='black', 

            marker='o', 

            s=35,

            alpha=0.5,

            label='Training data')

plt.scatter(y_test_pred,  

            y_test_pred - y_test, 

            c='skyblue', 

            marker='s', 

            s=35,

            alpha=0.7,

            label='Test data')



plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc='lower left')

plt.hlines(y=0, xmin=-50, xmax=1300, lw=2, color='red')

plt.xlim([-50, 1200])

plt.tight_layout()



plt.show()