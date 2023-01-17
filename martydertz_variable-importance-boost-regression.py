# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline  

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", index_col = 'Id')

test = pd.read_csv("../input/test.csv", index_col = 'Id')



######

# Preprocessing - log transform price and encode class vars

# Plotting Sales prices and log-sales prices

log_sale_price = np.log(train['SalePrice'])

train_features = train.drop('SalePrice', axis = 1)

train_features = train_features.select_dtypes(include=[np.number]) 

train_features = train_features.fillna(train_features.mean())

train_features = train_features.drop(['BsmtFinSF1', 'BsmtFinSF2'], axis = 1)            

from sklearn import ensemble

from sklearn.metrics import mean_squared_error

offset = int(train.shape[0] * 0.9)

X_train, y_train = train_features[:offset], log_sale_price[:offset]

X_test, y_test = train_features[offset:], log_sale_price[offset:]
log_sale_price.plot.hist()
###############################################################################

# Fit boost regression model

params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 1,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

mse_br = mean_squared_error(y_test, clf.predict(X_test))

print("Graidient Boost MSE: %.4f" % mse_br)
# Fit Kernel Ridge Regression

from sklearn.kernel_ridge import KernelRidge

kr = KernelRidge(kernel='linear', alpha = .01, 

                 coef0= 6)



kr.fit(X_train, y_train)

mse_kr = mean_squared_error(y_test, kr.predict(X_test))

print("Ridge Regression MSE: %.4F" % mse_kr)
# Plot training deviance for Boost Regression model

# compute test set deviance

import matplotlib.pyplot as plt

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)

    

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
# Plot feature importance

feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, train_features.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# Fit regression model with all observations

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(train_features, log_sale_price);