import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head()
train.columns
test = pd.read_csv('../input/test.csv')
print(test.shape)
test.columns
y_train = train.winPlacePerc
X_train = train.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis=1)
models = []
models.append(('LR', LinearRegression()))
models.append(('RIDGE', Ridge()))
models.append(('LASSO', Lasso()))
models.append(('ELN', ElasticNet()))
models.append(('DT', DecisionTreeRegressor()))
scoring = 'r2'
seed = 1000

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train,  cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
import pprint
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
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
# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
random_grid
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = DecisionTreeRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

print( rf_random.best_estimator_ )
print( rf_random.best_score_ )
print( rf_random.best_params_ )
rf = DecisionTreeRegressor(criterion='mse', max_depth=20, max_features='auto',
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=4,
           min_samples_split=10, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best').fit(X_train, y_train)
X_test = test.drop(['Id', 'groupId', 'matchId'], axis=1)
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
submission.winPlacePerc = rf.predict(X_test)
submission.head()
submission.to_csv('submission.csv', index=False)