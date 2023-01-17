# import necessary packages
import pandas as pd
import os, time
import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import make_scorer
# loading data
train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data = pd.read_csv('../input/test.csv', index_col='Id')
y = train_data['SalePrice']
train_index = train_data.index
test_index = test_data.index
print("The shape of training data:\n", train_data.shape)
print("\nThe basic info of training data:\n", train_data.info())

# missing value check
print("Missing values in training data:\n", train_data.isnull().sum().sort_values(ascending=False))
print("\nMissing values in test data:\n", test_data.isnull().sum().sort_values(ascending=False))

# handle missing value
df = pd.concat([train_data.drop('SalePrice', axis=1), test_data], axis=0)
df = df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) # delete the variables with too many missing values
del train_data, test_data
# fill in missing value with mode
for col in df.columns:
    if df[col].isnull().sum()!=0:
        df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum().sort_values(ascending=False) 
df.shape
# categorical variables
i=0
print("The number of categories for each categorical variable:")     
for col in df.columns:
    if df.dtypes[col] =='O':
        print(col, len(df[col].value_counts())  )
        i += 1
print('There are %d categorical variables.'%i)
# The easiest process of category, not recommand!
for col in df.columns:
    if df.dtypes[col] =='O':
        df.drop(col, axis=1, inplace=True)
# scale the data to mean 0 and std 1
train_data = pd.concat([df.loc[train_index,:],y], axis=1)
X = train_data.drop('SalePrice', axis=1)
test_data = df.loc[test_index,:]

scaler = preprocessing.StandardScaler()
scaler.fit(train_data.drop('SalePrice', axis=1))
train_data.iloc[:,:-1] = scaler.transform(train_data.drop('SalePrice', axis=1))
test_data_X = scaler.transform(test_data)
# define a score function according to the rule of competition
def sqrt_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_log_error(y_true, y_pred))
# split the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# a easy try of gridsearch to choose the hyperparameters of model
rf = RandomForestRegressor()
param_grid = {'n_estimators':[10,20,30], 'min_samples_split':[2],'min_samples_leaf':[1,3,5],
              }
grid = GridSearchCV(rf, param_grid = param_grid, cv=5, 
                    scoring=make_scorer(sqrt_mean_squared_log_error))
start = time.time()
grid.fit(X_train, y_train)
print("Training time:%0.4fs."%(time.time()-start))
# cv score for the best esimator
grid.best_score_
# the test score in training data
rf_tuned = grid.best_estimator_
rf_tuned.fit(X_train, y_train)
sqrt_mean_squared_log_error(y_test, rf_tuned.predict(X_test))

# predict and save the results for submission
rf_tuned.fit(X, y)
final = rf_tuned.predict(test_data_X)
sub = pd.DataFrame(data={'Id':test_data.index, 'SalePrice':final})
sub.to_csv("rf_submission_easy_try.csv", index=False)
