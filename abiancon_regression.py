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
dataset = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')

y = dataset['Apparent Temperature (C)']

dataset = dataset.drop(columns = ['Formatted Date', 'Loud Cover', 'Daily Summary','Apparent Temperature (C)'])

dataset.head()

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

dataset['Summary'] = encoder.fit_transform(dataset['Summary'])

column = dataset['Precip Type'].apply(str)

dataset['Precip Type'] = encoder.fit_transform(column)

dataset.head()
import seaborn as sns

import matplotlib.pyplot as plt

corrmat = dataset.corr()

plt.figure(figsize=(10,10))

g = sns.heatmap(corrmat,annot=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.3, random_state = 123)
from sklearn.preprocessing import StandardScaler

s = StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = s.fit_transform(X_train)

X_test = s.transform(X_test)
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score

mlp = MLPRegressor(hidden_layer_sizes = 5, activation = 'tanh', alpha = 0.0001)

mlp.fit(X_train, y_train)

y_predtest = mlp.predict(X_test)

print('TEST SET:')

print('MSE: ', mean_squared_error(y_test, y_predtest))

print('R2: ', r2_score(y_test, y_predtest))

from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LinearRegression, LassoCV, ElasticNetCV

lregr = LinearRegression()

lregr.fit(X_train, y_train)

y_predtest = lregr.predict(X_test)

print('TEST SET:')

print('MSE: ', mean_squared_error(y_test, y_predtest))

print('R2: ', r2_score(y_test, y_predtest))

print('Coefficienti:')

lregr.coef_
#è fondamentale calcolare il valore della penalità, dato dal coefficiente alfa in sklearn.

#Per farlo sarà calcolato alfa tra questi otto valori, tramite RidgeCV, che permette di adattare

#un modello di regressione ridge con cross-validation.



alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

model_cv = RidgeCV(alphas = alpha)

model_cv.fit(X_train, y_train)

print('ALPHA: ', model_cv.alpha_)



ridge_model = Ridge(alpha = 1.0)

ridge_model.fit(X_train, y_train)

y_predtest = ridge_model.predict(X_test)

print('TEST SET:')

print('MSE: ', mean_squared_error(y_test, y_predtest))

print('R2: ', r2_score(y_test, y_predtest))

print('Coefficienti:')

ridge_model.coef_
#calcolo di alpha tramite LassoCV



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

model_cv = LassoCV(alphas = alpha)

model_cv.fit(X_train, y_train)

print('ALPHA: ', model_cv.alpha_)



lasso_model = Lasso(alpha = 0.0001)

lasso_model.fit(X_train, y_train)

y_predtest = lasso_model.predict(X_test)

print('TEST SET:')

print('MSE: ', mean_squared_error(y_test, y_predtest))

print('R2: ', r2_score(y_test, y_predtest))

print('Coefficienti:')

lasso_model.coef_
#calcolo di alpha tramite ElasticNetCV



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

alpha = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

model_cv = ElasticNetCV(alphas = alpha)

model_cv.fit(X_train, y_train)

print('ALPHA: ', model_cv.alpha_)



elastic_model = ElasticNet(alpha = 0.0001)

elastic_model.fit(X_train, y_train)

y_predtest = elastic_model.predict(X_test)

print('TEST SET:')

print('MSE: ', mean_squared_error(y_test, y_predtest))

print('R2: ', r2_score(y_test, y_predtest))

print('Coefficienti:')

elastic_model.coef_