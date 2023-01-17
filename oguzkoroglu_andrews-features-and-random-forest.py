import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from gplearn.genetic import SymbolicRegressor,SymbolicTransformer

from gplearn.functions import make_function



from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import RFE

from sklearn.feature_selection import RFECV

from sklearn.model_selection import RandomizedSearchCV



import os



print(os.listdir("../input"))

print(os.listdir("../input/LANL-Earthquake-Prediction"))

print(os.listdir("../input/lanl-features"))
X = pd.read_csv('../input/lanl-features/train_features_denoised.csv')

X_test = pd.read_csv('../input/lanl-features/test_features_denoised.csv')

y = pd.read_csv('../input/lanl-features/y.csv')

submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv',index_col='seg_id')
X.drop('seg_id',axis=1,inplace=True)

X_test.drop('seg_id',axis=1,inplace=True)

X.drop('target',axis=1,inplace=True)

X_test.drop('target',axis=1,inplace=True)



alldata = pd.concat([X, X_test])



scaler = StandardScaler()



alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)



X = alldata[:X.shape[0]]

X_test = alldata[X.shape[0]:]
%%time

corr_matrix = X.corr()

corr_matrix = corr_matrix.abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



X = X.drop(to_drop, axis=1)

X_test = X_test.drop(to_drop, axis=1)

print(X.shape)

print(X_test.shape)
X["mean_y"] = np.full(len(y), y.values.mean())

X["max_y"] = np.full(len(y), y.values.max())

X["min_y"] = np.full(len(y), y.values.min())

X["std_y"] = np.full(len(y), y.values.std())



X_test["mean_y"] = np.full(len(X_test), y.values.mean())

X_test["max_y"] = np.full(len(X_test), y.values.max())

X_test["min_y"] = np.full(len(X_test), y.values.min())

X_test["std_y"] = np.full(len(X_test), y.values.std())



print(X.shape)

print(X_test.shape)
%%time

rf = RandomForestRegressor(n_estimators = 10)

rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1) #3-fold cross-validation with mae

rfecv = rfecv.fit(X, y.values)

print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X.columns[rfecv.support_])



X = X[X.columns[rfecv.support_].values]

X_test = X_test[X_test.columns[rfecv.support_].values]

print(X.shape)

print(X_test.shape)
print("Features:", list(X.columns))
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
print("Grid Search Parameters", random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X, y)
rf_random.best_params_
def evaluate(model, features=X, labels=y):

    predictions = model.predict(features)    

    mae=mean_absolute_error(labels, predictions)

    print('Model Performance')

    print('Mean Absolute Error: {:0.4f}.'.format(mae))

    return mae
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)

base_model.fit(X, y)

base_mae = evaluate(base_model, X, y)
best_random = rf_random.best_estimator_

random_mae = evaluate(best_random, X, y)
print('Improvement of {:0.2f}%.'.format(100 * (base_mae - random_mae) / base_mae))
submission.time_to_failure = best_random.predict(X_test)

submission.to_csv('submission.csv', index=True)
submission.head(10)