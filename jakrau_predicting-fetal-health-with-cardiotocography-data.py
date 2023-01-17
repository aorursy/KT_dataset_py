import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import imblearn

import copy

import seaborn as sns

import sklearn

from sklearn.inspection import permutation_importance

from sklearn.impute import KNNImputer

import lightgbm as lgb



# fix seed to make results reproducible

seed = 42
# Read data

df = pd.read_excel('../input/uci-cardiotocography/CTG.xls', header=0, sheet_name=2, skipfooter=3)

df.dropna(axis=0, thresh=10, inplace=True) # drop empty rows from the original xls file



# drop irrelevant and classification columns

df.drop(columns=['FileName', 'Date', 'SegFile', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP','CLASS', 'DR',], inplace=True)

df
# Make binary outcome variable (normal,suspect+pathological)

df['status'] = np.where(df.NSP == 1, -1, 1)  # recodes normal to -1 and everything else to 1



# Plot histogram

fig, ax = plt.subplots(1,1)

df.status.hist()

class_names = ['normal', 'suspect/pathologic']

ax.set_xticks([-1,1])

ax.set_xticklabels(class_names)
# Histogram for all features

df_X = df.drop(columns=['NSP', 'status'])

fig, ax = plt.subplots(1, 1, figsize=(15, 15))

df_X.hist(ax=ax)

plt.show()
print('Number of missing values: ', df.isnull().sum().sum())  # check for missing
# Boxplots for feature distributions

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

df_scale_X = pd.DataFrame(sklearn.preprocessing.scale(df_X), columns=df_X.columns)

df_scale_X.boxplot(ax=ax, rot=45)

plt.show()
# Feature correlation heatmap

fig, ax = plt.subplots(1, 1, figsize=(20, 15))

corr = df.drop(columns=['NSP']).corr()

corr = corr.round(decimals=2)

corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool)) # make heatmap lower triangular (remove redundant info)

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)

plt.xticks(rotation=45)

plt.show()
# shuffle data

df = df.sample(frac=1, random_state=seed)



# make vector of class labels and feature matrix

y, X = df.status.values, df.drop(columns=['NSP', 'status']).values.astype('float')
def zscore_outlier_removal(X, threshold=100):

    """ Sets feature values in X that are more than (threshold * feature standard deviation) away from feature mean

    to NaN. Returns X with original length but some column values are NaN. At default value 100, no outlier treatment occurs.

    """

    new_X = copy.deepcopy(X)

    new_X[abs(sklearn.preprocessing.scale(X)) > threshold] = np.nan



    return new_X



# Make zscore feature outlier removal a transformer function

zscore_outlier_removal = sklearn.preprocessing.FunctionTransformer(zscore_outlier_removal,

    kw_args=dict(threshold=7))





# Replace feature outliers with imputed values via KNN

KNN_impute = KNNImputer()
# Polynomial feature expansion

poly = sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)



# demean and scale to unit variance

scale = sklearn.preprocessing.StandardScaler()
def print_cv(cv, X, y, model_name):

    """ Prints best score, best parameter values, and in and out of sample confusion matrices for a cv

    result """

    print('Results for {}:'.format(model_name))

    print('The best out-of-sample performance is {}'.format(cv.best_score_))

    print('Best parameter values: ', cv.best_params_)



    pred = cv.best_estimator_.predict(X)

    print('In-sample confusion matrix of best estimator:\n{}'.format(sklearn.metrics.confusion_matrix(y, pred)))



    cross_val_pred = sklearn.model_selection.cross_val_predict(cv.best_estimator_, X, y, cv=5, n_jobs=-1)

    print('Out-of-sample confusion matrix of best estimator:\n{}'.format(

        sklearn.metrics.confusion_matrix(y, cross_val_pred)))



    return
# Support Vector Machine

svm = sklearn.svm.SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced', probability=True,

    decision_function_shape='ovr')
svm_pipe = sklearn.pipeline.Pipeline(

    [('outlier', zscore_outlier_removal), ('impute', KNN_impute), ('scale', scale), ('svm', svm)])
# values to try for cross-validation

zscore_threshold_vals = [100, 9, 8, 7, 6, 5, 4, 3]  # 100 = no outlier treatment

kernels = ['rbf', 'poly']

poly_degrees = [1, 2, 3] # degrees of polynomial expansion (only relevant for polynomial kernel)

penalty_vals = [np.e ** i for i in np.linspace(-3, 3, 8)]
# kernel and penalty search

svm_grid = {"svm__C": penalty_vals, "svm__kernel": kernels, "svm__degree": poly_degrees}



svm_cv = sklearn.model_selection.GridSearchCV(svm_pipe, svm_grid, scoring='balanced_accuracy', n_jobs=-1,

    refit=True, verbose=True, return_train_score=True, cv=5)

svm_cv.fit(X, y)

print_cv(svm_cv, X, y, 'support vector machine')
random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2,

    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=10,

    bootstrap=True, oob_score=False, n_jobs=-1, random_state=seed, class_weight='balanced')

# values for cross-validation

tree_numbers = [20, 50, 100, 200]

tree_depths = [1, 2, 3, 4, 5, 6, 7, 8]

max_feature_vals = [3, 5, 8, 10, 15]



random_forest_grid = {"n_estimators": tree_numbers,

    "max_depth": tree_depths,

    "max_features": max_feature_vals}



random_forest_cv = sklearn.model_selection.GridSearchCV(random_forest, random_forest_grid,

    scoring='balanced_accuracy', n_jobs=-1, refit=True, verbose=True, return_train_score=True, cv=5)

random_forest_cv.fit(X,y)



print_cv(random_forest_cv, X, y, 'Random forest')
gbm = lgb.LGBMClassifier(max_depth=5, class_weight='balanced',

                        learning_rate=1, n_estimators=500, random_state=seed)



# values to try for cross-validation

learning_rates = [0.01,0.1,1]

n_iterations_vals = [50,200,1000]

tree_depths = [1,2,4,8,16]



gbm_grid = {"max_depth": tree_depths, 

            "learning_rate": learning_rates, 

            "n_estimators": n_iterations_vals}



# stratified 5-fold cross-validation

cv_boost = sklearn.model_selection.GridSearchCV(gbm, gbm_grid, scoring='balanced_accuracy',

    n_jobs=-1, refit=True, verbose=True, return_train_score=False, cv=5)

cv_boost.fit(X,y)



print_cv(cv_boost, X, y, 'AdaBoost')
models = [('forest', random_forest_cv.best_estimator_), ('svm', svm_cv.best_estimator_),

    ('boost', cv_boost.best_estimator_)]



vote_clf = sklearn.ensemble.VotingClassifier(models)



# CV

vote_cv = sklearn.model_selection.GridSearchCV(vote_clf, {}, scoring='balanced_accuracy', n_jobs=-1, refit=True, cv=5)

vote_cv.fit(X,y)

print_cv(vote_cv, X, y, 'Voting ensemble')
stack = sklearn.ensemble.StackingClassifier(models, cv=5, stack_method='auto')



# CV

stack_cv = sklearn.model_selection.GridSearchCV(stack, {}, scoring='balanced_accuracy', n_jobs=-1, refit=True, cv=5)

stack_cv.fit(X,y)

print_cv(stack_cv, X, y, 'stacking ensemble')