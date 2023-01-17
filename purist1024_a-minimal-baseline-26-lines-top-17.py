import pandas as pd

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.model_selection import cross_val_predict



rawXy_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

y = np.log(rawXy_train.SalePrice)

rawX_train = rawXy_train.drop(columns=['SalePrice'])

numeric_columns = [col for col in rawX_train.columns if rawX_train[col].dtype != "object"]

def transform_X(X):

    X = X.fillna(0)

    X = X.join(np.sqrt(X[numeric_columns]), rsuffix="_SQRT")

    X = pd.get_dummies(X)

    return X

X_train = transform_X(rawX_train)



cv_errors = np.abs(cross_val_predict(BayesianRidge(), X_train, y, n_jobs=-1, cv=10) - y)

outliers = list(cv_errors[cv_errors > (np.mean(cv_errors) + 2*np.std(cv_errors))].index)

X_train = X_train.drop(outliers)

y = y.drop(outliers)

final_model = BayesianRidge().fit(X_train, y)



X_test = transform_X(pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id'))

X_test = pd.DataFrame({col: X_test.get(col, 0) for col in X_train.columns})

predictions = np.exp(final_model.predict(X_test))

pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions}).to_csv('baseline-submission.csv', index=False)