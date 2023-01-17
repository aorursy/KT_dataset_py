import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#the data comes from this website https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split



from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor



from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
cols = 'frequency attack_angle chord_length velocity thickness sound_level'.split()

df = pd.read_table('../input/airfoil_self_noise.dat', header = None, names=cols)
df.info()
df.describe()
X = df.iloc[:,0:5]

y = df.iloc[:,5:6]
X_train, X_test, y_train, y_test = train_test_split(X, y)
#the pair plot shows that linear or ridge regression may not be the best choices here

sns.pairplot(X)
#because I was learning and I wanted to show that how bad they performed, I ran the following two models

reg = Ridge(alpha = 0.01).fit(X_train, y_train)



print(f'The training set score of the model is {round(reg.score(X_train, y_train), 2)}')

print(f'The test set score of the model is {round(reg.score(X_test, y_test), 2)}')
reg = LinearRegression().fit(X_train, y_train)



print(f'The training set score of the model is {round(reg.score(X_train, y_train), 2)}')

print(f'The test set score of the model is {round(reg.score(X_test, y_test), 2)}')
reg = RandomForestRegressor(n_estimators = 1000, max_features = 'log2').fit(X_train, y_train.values.ravel())



print(f'The training set score of the model is {round(reg.score(X_train, y_train), 2)}')

print(f'The test set score of the model is {round(reg.score(X_test, y_test), 2)}')
reg = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 10000).fit(X_train, y_train.values.ravel())

    

print(f'The training set score of the model is {round(reg.score(X_train, y_train), 2)}')

print(f'The test set score of the model is {round(reg.score(X_test, y_test), 2)}')
#this was much faster than GradientBoostingRegressor from SKLearn

xgb_model = XGBRegressor(eta = 0.1, n_estimators = 1000).fit(X_train, y_train)



print(f'The training set score of the model is {round(xgb_model.score(X_train, y_train), 2)}')

print(f'The test set score of the model is {round(xgb_model.score(X_test, y_test), 2)}')
parameters_grid = {

    'n_estimators': [100, 1000, 10000],

    'learning_rate': [0.001, 0.1, 0.2]

}



reg = GridSearchCV(estimator = XGBRegressor(),

                   param_grid = parameters_grid,

                   cv = 10,

                   n_jobs = -1)



reg.fit(X_train, y_train)



#best score of accuracy

print('Best score:', reg.best_score_)



# View the best parameters for the model found using grid search

print('Best Number of Trees:',reg.best_estimator_.n_estimators) 

print('Best Learning Rate:',reg.best_estimator_.learning_rate)
xgb_model = XGBRegressor(learning_rate = reg.best_estimator_.learning_rate,

                         n_estimators = reg.best_estimator_.n_estimators).fit(X_train, y_train)



y_pred = xgb_model.predict(X_test)



print(f'The R2 Score is {r2_score(y_test, y_pred)}') #one of the best

print(f'The Explained Variance Score is {explained_variance_score(y_test, y_pred)}') #very good as well

print(f'The Mean Squared Error is {mean_squared_error(y_test, y_pred)}')

print(f'The Root Mean Squared Error is {np.sqrt(mean_squared_error(y_test, y_pred))}')
g = sns.distplot(y_pred, hist = False)

g = sns.distplot(y_test, hist = False)

plt.show()