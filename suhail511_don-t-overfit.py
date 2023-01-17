import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import GradientBoostingRegressor





train = pd.read_csv("../input/train.csv").drop('id', axis=1)



y_train = train['target']

X_train = train.drop('target', axis=1)



# test = pd.read_csv('../input/test.csv')

# X_test = test.drop('id', axis = 1)



# submission = pd.read_csv('../input/sample_submission.csv')



# clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear').fit(X_train, y_train)



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state = 40)
# clf = Ridge(alpha=5.0,normalize=True,copy_X=True,max_iter=None,tol=0.1,solver='svd').fit(X_train, y_train)
# submission['target'] = clf.predict(X_test)

# submission.to_csv('submission.csv', index=False)
# Takes in a model, trains the model, and evaluates the model on the test set

def fit_and_evaluate(model):

    

    # Train the model

    model.fit(X_train, y_train)

    

    # Make predictions and evalute

    model_pred = model.predict(X_test)

    model_auroc = roc_auc_score(y_test, model_pred)

    

    # Return the performance metric

    return model_auroc
gradient_boosted = GradientBoostingRegressor(random_state=60)

gradient_boosted_auroc = fit_and_evaluate(gradient_boosted)



print('Gradient Boosted Regression Performance on the test set: AUROC = %0.4f' % gradient_boosted_auroc)
lr = LinearRegression()

lr_auroc = fit_and_evaluate(lr)



print('Linear Regression Performance on the test set: AUROC = %0.4f' % lr_auroc)
svm = SVR(C = 1000, gamma = 0.1)

svm_auroc = fit_and_evaluate(svm)



print('Support Vector Machine Regression Performance on the test set: AUROC = %0.4f' % svm_auroc)
random_forest = RandomForestRegressor(random_state=60)

random_forest_auroc = fit_and_evaluate(random_forest)



print('Random Forest Regression Performance on the test set: AUROC = %0.4f' % random_forest_auroc)
knn = KNeighborsRegressor(n_neighbors=10)

knn_auroc = fit_and_evaluate(knn)



print('K-Nearest Neighbors Regression Performance on the test set: AUROC = %0.4f' % knn_auroc)
# Loss function to be optimized

loss = ['ls', 'lad', 'huber']



# Number of trees used in the boosting process

n_estimators = [1500,1700, 1900, 2100, 2300]



# Maximum depth of each tree

max_depth = [2, 3, 5, 10, 15]



# Minimum number of samples per leaf

min_samples_leaf = [2, 4, 6, 8]



# Minimum number of samples to split a node

min_samples_split = [2, 4, 6, 10, 13, 16]



# Maximum number of features to consider for making splits

max_features = ['auto', 'sqrt', 'log2', None]



# Define the grid of hyperparameters to search

hyperparameter_grid = {'loss': loss,

                       'n_estimators': n_estimators,

                       'max_depth': max_depth,

                       'min_samples_leaf': min_samples_leaf,

                       'min_samples_split': min_samples_split,

                       'max_features': max_features}
# Create the model to use for hyperparameter tuning

model = GradientBoostingRegressor(random_state = 42)



# Set up the random search with 4-fold cross validation

random_cv = RandomizedSearchCV(estimator=model,

                               param_distributions=hyperparameter_grid,

                               cv=4, n_iter=25, 

                               scoring = 'roc_auc',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)
random_cv.fit(X_train, y_train)
# # Get all of the cv results and sort by the test performance

# random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)



# random_results.head(10)
random_cv.best_estimator_
gradient_boosted = random_cv.best_estimator_

gradient_boosted_auroc = fit_and_evaluate(gradient_boosted)



print('Gradient Boosted Regression best_estimator_ Performance on the test set: AUROC = %0.4f' % gradient_boosted_auroc)
ridge_classifier = Ridge(alpha=5.0,

                         normalize=True,

                         copy_X=True,

                         max_iter=None,

                         tol=0.1,solver='svd')

ridge_classifier_auroc = fit_and_evaluate(ridge_classifier)

print('Ridge Performance on the test set: AUROC = %0.4f' % ridge_classifier_auroc)
alpha=[1.0,5,10.0,50,100,500,1000,5000,10000]

fit_intercept=[True, False]

normalize=[True, False]

copy_X=[True, False]

tol=[0.001,0.01,0.1,1,10]

solver=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

hyperparameter_grid = {'alpha':alpha,

                       'fit_intercept':fit_intercept,

                       'normalize':normalize,

                       'copy_X':copy_X,

                       'tol':tol,

                       'solver':solver

                      }
# Create the model to use for hyperparameter tuning

model = Ridge()



# Set up the random search with 4-fold cross validation

random_cv = RandomizedSearchCV(estimator=model,

                               param_distributions=hyperparameter_grid,

                               cv=4, n_iter=25, 

                               scoring = 'roc_auc',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)
random_cv.fit(X_train, y_train)
random_cv.best_estimator_
ridge_classifier = random_cv.best_estimator_

ridge_classifier_auroc = fit_and_evaluate(ridge_classifier)



print('ridge_classifier best_estimator_ Performance on the test set: AUROC = %0.4f' % ridge_classifier_auroc)
train = pd.read_csv("../input/train.csv").drop('id', axis=1)



y_train = train['target']

X_train = train.drop('target', axis=1)



test = pd.read_csv('../input/test.csv')

X_test = test.drop('id', axis = 1)



submission = pd.read_csv('../input/sample_submission.csv')
clf = Ridge(alpha=10000, copy_X=True, fit_intercept=True, max_iter=None,

   normalize=False, random_state=None, solver='sag', tol=0.001).fit(X_train, y_train)

submission['target'] = clf.predict(X_test)

submission.to_csv('submission.csv', index=False)