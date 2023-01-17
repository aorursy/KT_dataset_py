# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
np.random.seed(seed=42)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ids = test.Id

#prepare for analysis
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train,test = train.align(test,join='left', axis=1)

# from sklearn.preprocessing import Imputer
# imputer = Imputer()
# train = imputer.fit_transform(train)
# test = imputer.fit_transform(test)

train = train.dropna(axis=1, how='any') #drop missing values
test = test.dropna(axis=1, how='any')
print(train.shape,test.shape)

Y = train["SalePrice"]
X = train.drop(["SalePrice"], axis=1)
# print("Normalizing data...")
# from sklearn import preprocessing
# X = preprocessing.scale(X)
# test = preprocessing.scale(test)

print("PCA feature selection")
from sklearn.decomposition import PCA
pca = PCA(n_components=260)
X = pca.fit_transform(X)
test = pca.fit_transform(test)
print(X.shape,test.shape)
print("Linear regression")
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

# rms = sqrt(mean_squared_error(y_actual, y_predicted))

X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size=0.1,random_state=42)


reg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
fit = reg.fit(X_train,y_train)
y_predicted = reg.predict(X_val)
print('Train R2',reg.score(X_train,y_train))
print('Val R2', r2_score(y_val,y_predicted))
print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
print('Predicting on test set for submission...')
predictions=reg.predict(test)

print("Writing predictions...")
solution = pd.DataFrame({"id":ids, "SalePrice":predictions})
solution.to_csv("lin_reg.csv", index = False)
print("Ridge Regression")
from sklearn.linear_model import Ridge

reg = Ridge(alpha=0.1, fit_intercept=True, normalize=False, 
            copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=42)
#solver : {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}

fit = reg.fit(X_train,y_train)
y_predicted = reg.predict(X_val)

print('Train R2',reg.score(X_train,y_train))
print('Val R2', r2_score(y_val,y_predicted))
print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
print('Predicting on test set for submission...')
predictions=reg.predict(test)

print("Writing predictions...")
solution = pd.DataFrame({"id":ids, "SalePrice":predictions})
solution.to_csv("ridge_reg.csv", index = False)
print("Bayesian Ridge (winning model so far...)")
from sklearn.linear_model import BayesianRidge

reg = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, 
                    lambda_1=1e-06, lambda_2=1e-06, compute_score=True, fit_intercept=True, 
                    normalize=False, copy_X=True, verbose=True)

fit = reg.fit(X_train,y_train)
y_predicted = reg.predict(X_val)

print('Train R2',reg.score(X_train,y_train))
print('Val R2', r2_score(y_val,y_predicted))
print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
print('Predicting on test set for submission...')
predictions=reg.predict(test)

print("Writing predictions...")
solution = pd.DataFrame({"id":ids, "SalePrice":predictions})
solution.to_csv("bayes_ridge_reg.csv", index = False)
print("ElasticNet regression")
from sklearn.linear_model import ElasticNet

reg = ElasticNet(alpha=0.001, l1_ratio=0.8, fit_intercept=True, 
                 normalize=False, precompute=True, max_iter=1000000, copy_X=True,
                 tol=0.00001, warm_start=False, positive=False, random_state=42, selection='random')

fit = reg.fit(X_train,y_train)
y_predicted = reg.predict(X_val)

print('Train R2',reg.score(X_train,y_train))
print('Val R2', r2_score(y_val,y_predicted))
print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
print('Predicting on test set for submission...')
predictions=reg.predict(test)

print("Writing predictions...")
solution = pd.DataFrame({"id":ids, "SalePrice":predictions})
solution.to_csv("elasticnet_reg.csv", index = False)
print("Bayesian ARD Regression (too long and no inmediate improvements)")

# from sklearn.linear_model import ARDRegression

# reg = ARDRegression(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, 
#               lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=True, 
#               normalize=False, copy_X=True, verbose=True)

# fit = reg.fit(X_train,y_train)
# y_predicted = reg.predict(X_val)

# print('Train R2',reg.score(X_train,y_train))
# print('Val R2', r2_score(y_val,y_predicted))
# print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
# print('Predicting on test set for submission...')
# predictions=reg.predict(test)

# print("Writing predictions...")
# solution = pd.DataFrame({"id":ids, "SalePrice":predictions})
# solution.to_csv("elasticnet_reg.csv", index = False)
print("Simple NN")
print(X_train.shape)
from keras.models import Sequential
from keras.layers import Dense,Dropout
from matplotlib import pyplot as plt 
from keras import backend

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# create model
model = Sequential()
# model.add(Dropout(0.2, input_shape=(X_train.shape[1],)))
model.add(Dense(260, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=[rmse])
# train model
history = model.fit(X_train, y_train, epochs=2000, batch_size=8, verbose=2, validation_data=[X_val,y_val])
# plot metrics
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.legend(['train', 'val'], loc='upper left')
plt.show()

y_predicted = model.predict(X_val)
print('Val R2', r2_score(y_val,y_predicted))
print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
print('Predicting on test set for submission...')
predictions=model.predict(test)
predictions = predictions[:, 0]

print("Writing predictions...")
solution = pd.DataFrame({"id":ids, "SalePrice":predictions})
solution.to_csv("nn.csv", index = False)



