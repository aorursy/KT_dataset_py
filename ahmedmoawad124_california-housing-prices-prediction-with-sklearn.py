import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor



import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
housing_dataset = pd.read_csv('../input/california-housing-prices/housing.csv')
# Showing the first 5 columns 

housing_dataset.head()
# Showing random 5 samples

housing_dataset.sample(5)
# the shape of the data

housing_dataset.shape
housing_dataset.isna().sum()
# drop the missing data

housing_dataset = housing_dataset.dropna()



# the shape after dropping the missing data

housing_dataset.shape
housing_dataset.isna().sum()
# Exporing the categorical data

housing_dataset['ocean_proximity'].unique()
# Converting categorical values to numeric values using one-hot encoding

housing_dataset = pd.get_dummies(housing_dataset, columns= ['ocean_proximity'])



# Another techinque:

'''

ocean_proximity = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']

label_encoding = preprocessing.LabelEncoder()

label_encoding = label_encoding.fit(ocean_proximity)

housing_dataset['ocean_proximity'] = label_encoding.transform(housing_dataset['ocean_proximity'])

label_encoding.classes_

'''



# Showing the data after Converting categorical values to numeric values

housing_dataset.head()
# Original data frame had 10 columns, we now have 14 columns

housing_dataset.shape
# Showing the correlation between data

housing_dataset_correlation = housing_dataset.corr()

housing_dataset_correlation
# housing dataset correlation in heat map

plt.figure(figsize=(15,12))

sns.heatmap(housing_dataset_correlation, annot = True)
# Extracting the data

X = housing_dataset.drop('median_house_value', axis = 1)  # Features

Y = housing_dataset['median_house_value']                 # Target
# Splitting the data into traing and testing data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=45)
linear_regressor = LinearRegression(normalize = True, fit_intercept = False, copy_X = True).fit(x_train, y_train)

# Normailzation scales all numeric features to be between 0 and 1. 

# Having features in the same scale can vastly improve tne performance of your ML model
print("Training score : ", linear_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = linear_regressor.predict(x_test)



print("testing score : ", r2_score(y_test, y_pred))
linear_regressor.intercept_ , linear_regressor.coef_
# Hyper-Parameters Tuning

'''

linear_regressor_parameter = {'normalize': [True, False], 'fit_intercept': [True, False]}

linear_regressor_grid_search = GridSearchCV(LinearRegression(), linear_regressor_parameter, cv = 2)

linear_regressor_grid_search.fit(X, Y)

print('The best score',linear_regressor_grid_search.best_score_)

print('The best parameters',linear_regressor_grid_search.best_params_)

'''



# The result:



# The best parameters {'fit_intercept': False, 'normalize': True}
lasso_regressor = Lasso(alpha = 1, fit_intercept= True, normalize= False, max_iter = 20000).fit(x_train, y_train)
print("Training score : ", lasso_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = lasso_regressor.predict(x_test)

print("testing score : ", r2_score(y_test, y_pred))
lasso_regressor.intercept_ , lasso_regressor.coef_ , lasso_regressor. n_iter_
# hyper-Parameters Tuning

'''

lasso_regressor_parameter = {'alpha': [0.2,0.4,0.6,0.8,1], 'normalize': [True, False], 'fit_intercept': [True, False]}

lasso_regressor_grid_search = GridSearchCV(Lasso(max_iter = 400000), lasso_regressor_parameter, cv = 2)

lasso_regressor_grid_search.fit(X, Y)

print('The best score',lasso_regressor_grid_search.best_score_)

print('The best parameters',lasso_regressor_grid_search.best_params_)

'''



# The result:



# The best parameters {'alpha': 1, 'fit_intercept': True, 'normalize': False}
ridge_regressor = Ridge(alpha = 0.4, fit_intercept= True, normalize= False, max_iter = 20000).fit(x_train, y_train)
print("Training score : ", ridge_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = ridge_regressor.predict(x_test)

print("testing score : ", r2_score(y_test, y_pred))
ridge_regressor.intercept_ , ridge_regressor.coef_ , ridge_regressor. n_iter_
# hyper-Parameters Tuning

'''

ridge_regressor_parameter = {'alpha': [0.2,0.4,0.6,0.8,1], 'normalize': [True, False], 'fit_intercept': [True, False]}

ridge_regressor_grid_search = GridSearchCV(Ridge(max_iter = 400000), ridge_regressor_parameter, cv = 2)

ridge_regressor_grid_search.fit(X, Y)

print('The best score',ridge_regressor_grid_search.best_score_)

print('The best parameters',ridge_regressor_grid_search.best_params_)

'''



# The result:



# The best parameters {'alpha': 0.4, 'fit_intercept': True, 'normalize': False}
elastic_regressor = ElasticNet(alpha = 1, l1_ratio = 1, normalize = False, fit_intercept= True, max_iter = 20000).fit(x_train, y_train)
print("Training score : ", elastic_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = elastic_regressor.predict(x_test)

print("testing score : ", r2_score(y_test, y_pred))
elastic_regressor.intercept_ , elastic_regressor.coef_ , elastic_regressor. n_iter_
# hyper-Parameters Tuning

'''

elastic_regressor_parameter = {'alpha': [0.2,0.4,0.6,0.8,1], 'l1_ratio': [0,0.2,0.5,0.8,1] , 'normalize': [True, False], 'fit_intercept': [True, False]}

elastic_regressor_grid_search = GridSearchCV(ElasticNet(max_iter = 400000), elastic_regressor_parameter, cv = 2)

elastic_regressor_grid_search.fit(X, Y)

print('The best score',elastic_regressor_grid_search.best_score_)

print('The best parameters',elastic_regressor_grid_search.best_params_)

'''



# The result:



# The best parameters {'alpha': 1, 'fit_intercept': True, 'l1_ratio': 1, 'normalize': False}
# SVR tries to fit as many points as possiple into a margine surrounding the best fit line

svr_regressor = SVR(kernel='linear', epsilon = 0.2, C = 1).fit(x_train, y_train)
print("Training score : ", svr_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = svr_regressor.predict(x_test)

print("testing score : ", r2_score(y_test, y_pred))
# hyper-Parameters Tuning

'''

svr_regressor_parameter = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'epsilon': [0.05,0.1,0.2,0.3], 'C': [0.2,0.5,0.8,1]}

svr_regressor_grid_search = GridSearchCV(SVR(), svr_regressor_parameter, cv = 2)

svr_regressor_grid_search.fit(X, Y)

print('The best score',svr_regressor_grid_search.best_score_)

print('The best parameters',svr_regressor_grid_search.best_params_)

'''

# Nearest Nieghbors regression uses training data to find what is most similar to the current sample

# Average y-values of K nearest nieghbors



knn_regressor = KNeighborsRegressor(n_neighbors = 10).fit(x_train, y_train)
print("Training score : ", knn_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = knn_regressor.predict(x_test)

print("testing score : ", r2_score(y_test, y_pred))
# hyper-Parameters Tuning

'''

knn_regressor_parameter = {'n_neighbors': [3,5,8,10,15,20,25]}

knn_regressor_grid_search = GridSearchCV(KNeighborsRegressor(), knn_regressor_parameter, cv = 2)

knn_regressor_grid_search.fit(X, Y)

print('The best score',knn_regressor_grid_search.best_score_)

print('The best parameters',knn_regressor_grid_search.best_params_)

'''
# Decision trees set up a tree structure on training data which helps make decisions based on rules



tree_regressor = DecisionTreeRegressor(max_depth = 7).fit(x_train, y_train)
print("Training score : ", tree_regressor.score(x_train, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = tree_regressor.predict(x_test)

print("testing score : ", r2_score(y_test, y_pred))
# hyper-Parameters Tuning

'''

treen_regressor_parameter = {'max_depth': [2,3,4,5,6,7,8,9,10]}

tree_regressor_grid_search = GridSearchCV(DecisionTreeRegressor(), tree_regressor_parameter, cv = 2)

tree_regressor_grid_search.fit(X, Y)

print('The best score',tree_regressor_grid_search.best_score_)

print('The best parameters',tree_regressor_grid_search.best_params_)

'''
# This is an iterative model where you use multiple iteration to find 

# the best linear model that fit your underlyning data.



# It works well with standarized features

# I will standarize the all features except the categorical features

features_not_categorical_columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']

features_not_categorical_train = x_train[features_not_categorical_columns]

x_train_scaled = x_train.copy()

scaler = StandardScaler()

features_not_categorical_train_scaled = scaler.fit_transform(features_not_categorical_train)

x_train_scaled[features_not_categorical_columns] = features_not_categorical_train_scaled



sgd_regressor = SGDRegressor(max_iter = 100000, tol = 1e-3).fit(x_train_scaled, y_train)
print("Training score : ", sgd_regressor.score(x_train_scaled, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
features_not_categorical_test = x_test[features_not_categorical_columns]

x_test_scaled = x_test.copy()

scaler = StandardScaler()

features_not_categorical_test_scaled = scaler.fit_transform(features_not_categorical_test)

x_test_scaled[features_not_categorical_columns] = features_not_categorical_test_scaled



y_pred = sgd_regressor.predict(x_test_scaled)

print("testing score : ", r2_score(y_test, y_pred))
# It do well with standarized features



nn_regressor = MLPRegressor(activation = 'relu', hidden_layer_sizes = (32,64,128,64,8), solver= 'lbfgs', max_iter= 20000).fit(x_train_scaled, y_train)



# 1- hidden_layer_sizes = (No. of units or neurons in 1st hidden layer, No. of units or neurons in 2nd hidden layer, .... )

# 2- Activation function for the hidden layer: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’  

# 3- solver for weight optimization : {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’

# 4- learning_rate_initdouble, default=0.001, The initial learning rate used. 

# It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.
print("Training score : ", nn_regressor.score(x_train_scaled, y_train))

# R-Square is a measure of how well our linear mogel captures the underlying variation in our training data
y_pred = nn_regressor.predict(x_test_scaled)

print("testing score : ", r2_score(y_test, y_pred))