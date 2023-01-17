# Importing main packages and settings

import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

pd.set_option('display.max_columns', 50)
# Loading the training dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# first view of the training data set

df_train.head()
# additional information about the training data set

print(df_train.info())

print(df_train.dtypes)
# analysis of the object features of the training data set

object_features = df_train.select_dtypes(include=[np.object])

object_features.describe()
# analysis of the object features of the test data set

# note the different number of unique values compared to the training set

object_features_test = df_test.select_dtypes(include=[np.object])

object_features_test.describe()
# analysis of the numerical features of the training data set

numeric_features = df_train.select_dtypes(include=[np.number])

numeric_features.describe()
# turning object features into dummy variables

df_train_dummies = pd.get_dummies(df_train, drop_first=True)

df_test_dummies = pd.get_dummies(df_test, drop_first=True)



# dropping ID and the target variable

df_train_dummies = df_train_dummies.drop(['ID','y'], axis=1)

df_test_dummies = df_test_dummies.drop('ID', axis=1)



print("Clean Train DataFrame With Dummy Variables: {}".format(df_train_dummies.shape))

print("Clean Test DataFrame With Dummy Variables: {}".format(df_test_dummies.shape))
# concatenate to only include columns in both data sets

# the number should be based on the number of columns. Original is 30471. Now set to 15471 after outlier handling etc.

df_temp = pd.concat([df_train_dummies, df_test_dummies], join='inner')

df_temp_train = df_temp[:len(df_train.index)]

df_temp_test = df_temp[len(df_train.index):]



# check shapes of combined df and split out again

print(df_temp.shape)

print(df_temp_train.shape)

print(df_temp_test.shape)
# defining X and y

X = df_temp_train

test_X = df_temp_test

y = df_train['y']
# Import the relevant sklearn packages

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_regression

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV

from sklearn.metrics import mean_squared_error
# instantiating

gbr = GradientBoostingRegressor()



# setting up steps for the pipeline, with and without imputating

steps = [('GradientBoostingRegressor', gbr)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# Compute 5-fold cross-validation scores: cv_scores

cv_scores_dummies = cross_val_score(pipe, X, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores_dummies)



print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_dummies)))
# Fitting a feature selector

def feature_selection(data):

    selector = VarianceThreshold(.98 * (1 - .98))

    selector.fit(data)

    return selector

 

#Learn the features to filter from train set

fs = feature_selection(X)

 

#Transform train and test subsets

X_transformed = fs.transform(X)

test_X_transformed = fs.transform(test_X)



print(X_transformed.shape)

print(test_X_transformed.shape)
# instantiating

gbr = GradientBoostingRegressor()



# setting up steps for the pipeline, with and without imputating

steps = [('GradientBoostingRegressor', gbr)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_transformed_train, X_transformed_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_transformed_train, y_train)

y_pred = pipe.predict(X_transformed_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_transformed_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# Compute 5-fold cross-validation scores: cv_scores

cv_scores_dummies = cross_val_score(pipe, X_transformed, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores_dummies)



print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_dummies)))
skb = SelectKBest(f_regression, k=50)



#Learn the features to filter from train set

skb.fit(X, y)



# transform the data sets

X_transformed_kbest = skb.transform(X)

test_X_transformed_kbest = skb.transform(test_X)



print(X_transformed_kbest.shape)

print(test_X_transformed_kbest.shape)
# instantiating

las = Lasso(alpha=0.1)



# setting up steps for the pipeline, with and without imputating

steps = [('Lasso', las)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# initiating

lscv = LassoCV()



# setting up steps for the pipeline, with and without imputating

steps = [('LassoCV', lscv)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# instantiating

rid = Ridge()



# setting up steps for the pipeline, with and without imputating

steps = [('Ridge', rid)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# instantiating

rcv = RidgeCV()



# setting up steps for the pipeline, with and without imputating

steps = [('RidgeCV', rcv)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# instantiating

els = ElasticNet()



# setting up steps for the pipeline, with and without imputating

steps = [('ElasticNet', els)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
# instantiating

elcv = ElasticNetCV()



# setting up steps for the pipeline, with and without imputating

steps = [('ElasticNetCV', elcv)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train ang test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
df_columns = df_train.columns



# Instantiate a lasso regressor: lasso

lasso = Lasso(alpha=0.4, normalize=True)



# Fit the regressor to the data

lasso.fit(X, y)



# Compute and print the coefficients

lasso_coef = lasso.coef_

# print(lasso_coef)



# Plot the coefficients

plt.plot(range(len(df_columns)), lasso_coef)

plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)

plt.margins(0.02)

plt.show()
X1 = df_train.drop(['ID', 'y'], axis=1)

X1 = X.select_dtypes(include=[np.number])
X1_test = df_test.drop(['ID'], axis=1)

X1_test = X1_test.select_dtypes(include=[np.number])
y = df_train['y']
from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression
# Fitting a feature selector

def feature_selection(data):

    selector = VarianceThreshold(.95 * (1 - .95))

    selector.fit(data)

    return selector

 

#Learn the features to filter from train set

fs = feature_selection(X1)

 

#Transform train and test subsets

X1_transformed = fs.transform(X1)

X1_test_transformed = fs.transform(X1_test)



print(X1_transformed.shape)

print(X1_test_transformed.shape)
# Fitting a feature selector

def feature_selection(data):

    selector = VarianceThreshold(.95 * (1 - .95))

    selector.fit(data)

    return selector

 

#Learn the features to filter from train set

fs = feature_selection(X1)

 

#Transform train and test subsets

X1_transformed = fs.transform(X1)

X1_test_transformed = fs.transform(X1_test)



print(X1_transformed.shape)

print(X1_test_transformed.shape)
skb = SelectKBest(f_regression, k=30)



skb.fit(X1_transformed, y)

X1_transformed_kbest = skb.transform(X1_transformed)

X1_test_transformed_kbest = skb.transform(X1_test_transformed)



print(X1_transformed_kbest.shape)

print(X1_test_transformed_kbest.shape)