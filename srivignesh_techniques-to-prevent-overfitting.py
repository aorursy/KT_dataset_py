import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import numpy as np 

import pandas as pd 

import xgboost



preprocessed_train = pd.read_csv('../input/preprocessed-train-data/preprocessed_train_data.csv')

preprocessed_test = pd.read_csv('../input/preprocessed-test-data/preprocessed_test_data.csv')

'''Split the data for training and testing'''

x_train, y_train = preprocessed_train[preprocessed_train.columns[:-1]], preprocessed_train['SalePrice']
def cross_validate(preprocessed_train, model):

    '''5-Fold Cross Validation'''

    kfold = KFold(n_splits=5, random_state=42)

    print(f'Model: {model.__class__.__name__}',end='\n')

    mse_errors = []

    for i, (train_idx, test_idx) in enumerate(kfold.split(preprocessed_train)):

        X_train, Y_train = x_train.loc[train_idx,:], y_train.loc[train_idx]

        X_test, Y_test = x_train.loc[test_idx,:], y_train.loc[test_idx]

        '''Fit the model'''

        model.fit(X_train,Y_train)

        '''Compute the predictions'''

        predictions = model.predict(X_test)

        '''Calculate Mean Squared Error and append to a list'''

        errors = mean_squared_error(Y_test, predictions)

        mse_errors.append(errors)

        print(f'Fold:{i+1}, MSE:{errors}',end='\n')

    return mse_errors



model = LinearRegression()

print(f'Variance of LinearRegression: {np.var(cross_validate(preprocessed_train, model))}',end='\n\n')



model = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)

mse_errors = cross_validate(preprocessed_train, model)

'''Compute the variance of XGBoost model'''

variance = np.var(mse_errors) 

'''

The XGBoost model's variance would be low when compared to Linear Regression so 

XGBoost has low variance and can generalize well.

'''

print(f'Variance of XGBRegressor: {variance}')
lasso = Lasso(alpha = 0.4)

mse_errors_lasso = cross_validate(preprocessed_train, lasso)

variance_lasso = np.var(mse_errors_lasso)

'''The variance is much less when compared to Linear Regression'''

print(f'Variance of LASSO: {variance_lasso}')
ridge = Ridge(alpha = 0.4)

mse_errors_ridge = cross_validate(preprocessed_train, ridge)

variance_ridge = np.var(mse_errors_ridge)

'''The variance is much less when compared to Linear Regression'''

print(f'Variance of Ridge: {variance_ridge}')
elasticnet = ElasticNet(alpha= 0.4, l1_ratio= 0.9)

mse_errors_en = cross_validate(preprocessed_train, elasticnet)
'''Kernel initializer denotes the distribution in which the weights of the neural networks are initialized'''

model = tf.keras.Sequential([

    tf.keras.layers.Dense(128, kernel_initializer='normal', activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(256, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(384, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dense(384, kernel_initializer='normal',activation='relu'),

    tf.keras.layers.Dropout(0.6),

    tf.keras.layers.Dense(1, kernel_initializer='normal',activation='linear')

])



msle = tf.keras.losses.MeanSquaredLogarithmicError()

model.compile(loss= msle, optimizer='adam', metrics=[msle])

model.fit(x_train.values, y_train.values, epochs=15, batch_size=64, validation_split = 0.2)