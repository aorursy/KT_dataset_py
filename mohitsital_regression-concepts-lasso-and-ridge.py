# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# thanks to : http://ethen8181.github.io/machine-learning/regularization/regularization.html

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#boston = pd.read_csv('/kaggle/input/bostonhoustingmlnd/housing.csv')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import Lasso, LassoCV

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics

# extract input and response variables (housing prices), 

# thanks to : http://ethen8181.github.io/machine-learning/regularization/regularization.html

# meaning of each variable is in the link above

feature_num = 20

boston = load_boston()

X = boston.data[:, :feature_num]

y = boston.target

features = boston.feature_names[:feature_num]

pd.DataFrame(X, columns = features).head()
# split into training and testing sets and standardize them

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

std = StandardScaler()

X_train_std = std.fit_transform(X_train)

X_test_std = std.transform(X_test)
# loop through different penalty score (alpha) and obtain the estimated coefficient (weights)

alphas = 10 ** np.arange(1, 5)

print('different alpha values:', alphas)



# stores the weights of each feature

ridge_weight = []

for alpha in alphas:    

    ridge = Ridge(alpha = alpha, fit_intercept = True)

    ridge.fit(X_train_std, y_train)

    ridge_weight.append(ridge.coef_)
def weight_versus_alpha_plot(weight, alphas, features):

    """

    Pass in the estimated weight, the alpha value and the names

    for the features and plot the model's estimated coefficient weight 

    for different alpha values

    """

    fig = plt.figure(figsize = (8, 6))

    

    # ensure that the weight is an array

    weight = np.array(weight)

    for col in range(weight.shape[1]):

        plt.plot(alphas, weight[:, col], label = features[col])



    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)

    

    # manually specify the coordinate of the legend

    plt.legend(bbox_to_anchor = (1.3, 0.9))

    plt.title('Coefficient Weight as Alpha Grows')

    plt.ylabel('Coefficient weight')

    plt.xlabel('alpha')

    return fig

# change default figure and font size

plt.rcParams['figure.figsize'] = 8, 6 

plt.rcParams['font.size'] = 12





ridge_fig = weight_versus_alpha_plot(ridge_weight, alphas, features)
# does the same thing above except for lasso

alphas = [0.01, 0.1, 1, 5, 8]

print('different alpha values:', alphas)



lasso_weight = []

for alpha in alphas:    

    lasso = Lasso(alpha = alpha, fit_intercept = True)

    lasso.fit(X_train_std, y_train)

    lasso_weight.append(lasso.coef_)



lasso_fig = weight_versus_alpha_plot(lasso_weight, alphas, features)
# alpha: array of alpha values to try; must be positive, increase for more regularization

# create an array of alpha values and select the best one with RidgeCV

alpha_range = 10. ** np.arange(-2, 3)

ridge_cv = RidgeCV(alphas = alpha_range, fit_intercept = True)

ridge_cv.fit(X_train_std, y_train)



# examine the coefficients and the errors of the predictions 

# using the best alpha value

y_pred = ridge_cv.predict(X_test_std)

print('coefficients:\n', ridge_cv.coef_)

print('best alpha:\n' , ridge_cv.alpha_)

print('\nRSS:', np.sum((y_test - y_pred) ** 2))

print(metrics.r2_score(y_test, y_pred))

from sklearn.metrics import r2_score



# n_alphas: number of alpha values (automatically chosen) to try

# select the best alpha with LassoCV

lasso_cv = LassoCV(n_alphas = 10, fit_intercept = True)

lasso_cv.fit(X_train_std, y_train)



# examine the coefficients and the errors of the predictions 

# using the best alpha value

y_pred = lasso_cv.predict(X_test_std)

print('coefficients:\n', lasso_cv.coef_)

print('best alpha:\n', lasso_cv.alpha_)

print('\nRSS:', np.sum(( y_test - y_pred ) ** 2))

print(r2_score(y_test, y_pred))
