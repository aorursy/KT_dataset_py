from sklearn.datasets import load_boston

import pandas as pd



boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

boston.head()
import matplotlib.pyplot as plt





b_corr = pd.DataFrame(boston.corr()['MEDV'])



b_corr_weak = b_corr[(abs(b_corr['MEDV']) > 0.2) & (abs(b_corr['MEDV']) <= 0.5)]

b_corr_weak = b_corr_weak.rename(columns={'MEDV': 'Слабая корреляция'})



b_corr_average = b_corr[(abs(b_corr['MEDV']) > 0.5) & (abs(b_corr['MEDV']) <= 0.7)]

b_corr_average = b_corr_average.rename(columns={'MEDV': 'Средняя корреляция'})



b_corr_strong = b_corr[(abs(b_corr['MEDV']) > 0.7) & (abs(b_corr['MEDV']) <= 0.9)]

b_corr_strong = b_corr_strong.rename(columns={'MEDV': 'Сильная корреляция'})



b_corr_vstrong = b_corr[(abs(b_corr['MEDV']) > 0.9) & (abs(b_corr['MEDV']) <= 1)]

b_corr_vstrong = b_corr_vstrong.rename(columns={'MEDV': 'Очень сильная корреляция'})



[b_corr_weak,b_corr_average, b_corr_strong, b_corr_vstrong]
from sklearn.model_selection import train_test_split



X = boston[['LSTAT']].values

y = boston['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40) #разделяем выборку на обучающую и тестовую
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

mean_squared_error(y_test, lr.predict(X_test)), r2_score(y_test, lr.predict(X_test))
from sklearn.preprocessing import PolynomialFeatures

import numpy as np





poly_quadric = PolynomialFeatures(degree=2)

poly_cubic = PolynomialFeatures(degree=3)

X_quad = poly_quadric.fit_transform(X_train)

X_cubic = poly_cubic.fit_transform(X_train)

X_quad_t = poly_quadric.fit_transform(X_test)

X_cubic_t = poly_cubic.fit_transform(X_test)

poly_cubic.fit(X_train, y_train)

poly_quadric.fit(X_train, y_train)



X_fit = np.arange(X_train.min(), X_train.max())[:, np.newaxis]



regr = lr.fit(X_train, y_train)

y_lin_fit = regr.predict(X_fit)

linear_r2 = r2_score(y_test, regr.predict(X_test))



regr = lr.fit(X_quad, y_train)

y_quad_fit = regr.predict(poly_quadric.fit_transform(X_fit))

quadratic_r2 = r2_score(y_test, regr.predict(X_quad_t))



regr = lr.fit(X_cubic, y_train)

y_cubic_fit = regr.predict(poly_cubic.fit_transform(X_fit))

cubic_r2 = r2_score(y_test, regr.predict(X_cubic_t))



# plot results

plt.scatter(X_train, y_train, label='training points', color='lightgray')



plt.plot(X_fit, y_lin_fit, 

         label='linear (d=1), $R^2={:.2f}$'.format(linear_r2), 

         color='blue', 

         lw=2, 

         linestyle=':')



plt.plot(X_fit, y_quad_fit, 

         label='quadratic (d=2), $R^2={:.2f}$'.format(quadratic_r2),

         color='red', 

         lw=2,

         linestyle='-')



plt.plot(X_fit, y_cubic_fit, 

         label='cubic (d=3), $R^2={:.2f}$'.format(cubic_r2),

         color='green', 

         lw=2, 

         linestyle='--')



plt.xlabel('% lower status of the population [LSTAT]')

plt.ylabel('Price in $1000\'s [MEDV]')

plt.legend(loc='upper right')

from sklearn.tree import DecisionTreeRegressor



dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train)



regr = dtr.fit(X_train, y_train)

y_dtr_fit = regr.predict(X_fit)

dtr_r2 = r2_score(y_test, regr.predict(X_test))

plt.scatter(X_train, y_train, label='training points', color='lightgray')

plt.plot(X_fit, y_dtr_fit, 

         label='decision tree , $R^2={:.2f}$'.format(dtr_r2),

         color='green', 

         lw=2, 

         linestyle='--')



plt.xlabel('% lower status of the population [LSTAT]')

plt.ylabel('Price in $1000\'s [MEDV]')

plt.legend(loc='upper right')
from sklearn import svm

from sklearn import preprocessing

from sklearn import utils



clf = svm.SVR()



regr = clf.fit(X_train, y_train)

y_clf_fit = regr.predict(X_fit)

clf_r2 = r2_score(y_test, regr.predict(X_test))

plt.scatter(X_train, y_train, label='training points', color='lightgray')

plt.plot(X_fit, y_clf_fit, 

         label='svm , $R^2={:.2f}$'.format(clf_r2),

         color='green', 

         lw=2, 

         linestyle='--')



plt.xlabel('% lower status of the population [LSTAT]')

plt.ylabel('Price in $1000\'s [MEDV]')

plt.legend(loc='upper right')
X = boston[['LSTAT', 'RM']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

X_fit = np.arange(X_train.min(), X_train.max())[:, np.newaxis]
regr = clf.fit(X_train, y_train)

y_clf_fit = regr.predict(X_test)

clf_r2 = r2_score(y_test, regr.predict(X_test))

clf_r2