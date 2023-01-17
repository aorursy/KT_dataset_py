import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, model_selection
import random
"""
Task: load the data into a variable named data
"""
file_name = '../input/SAheart.data'
data = pd.read_csv(file_name, sep=',', index_col=0)
data.head()
data['famhist_true'] = data['famhist'] == 'Present'
data['famhist_false'] = data['famhist'] == 'Absent'
data = data.drop(['famhist'], axis=1)
data.head()
def split(data):
    # control randomization for reproducibility
    np.random.seed(42)
    random.seed(42)
    train, test = model_selection.train_test_split(data)
    x_train = train.loc[:, train.columns != 'chd']
    y_train = train['chd']
    x_test = test.loc[:, test.columns != 'chd']
    y_test = test['chd']
    return x_train, y_train, x_test, y_test
def plot_feature(data, feature_name):
    plt.figure(figsize=(10, 3))
    plt.scatter(data[feature_name], data['chd'])
    plt.xlabel(feature_name)
    plt.ylabel('chd')
    plt.show()
"""
Feature list:
sbp tobacco ldl adiposity famhist_true famhist_false typea obesity alcohol age
"""
plot_feature(data, 'tobacco')
def evaluate(model, x_train, y_train, x_test, y_test):
    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)
    train_acc = metrics.accuracy_score(y_train, train_preds)
    test_acc = metrics.accuracy_score(y_test, test_preds)
    print('Train accuracy: %s' % train_acc)
    print('Test accuracy: %s' % test_acc)
def split_train_evaluate(model, data):
    x_train, y_train, x_test, y_test = split(data)
    model.fit(x_train, y_train)
    evaluate(model, x_train, y_train, x_test, y_test)

# randomly pick some maybe reasonable hyperparameters
model_bl = linear_model.SGDClassifier(
    loss='log', alpha=0.1, max_iter=1000, tol=-np.inf, class_weight='balanced')
split_train_evaluate(model_bl, data)
print('Sick: %s' % len(data[data['chd'] == True]))
print('Healthy: %s' % len(data[data['chd'] == False]))
302 / (160 + 302)
# this may take a few seconds to run
x_train, y_train, x_test, y_test = split(data)
grid_search = model_selection.GridSearchCV(
    estimator=linear_model.SGDClassifier(loss='log', tol=-np.inf, class_weight='balanced'),
    param_grid={'alpha': [0.1, 0.3],  # This is a tiny an very incomplete search space
                'max_iter': [10000, 15000]},  # These may be way too large?
    cv=10,
    return_train_score=True)
grid_search.fit(x_train, y_train)
r = pd.DataFrame(grid_search.cv_results_)
# we only want a subset of the columns for a precise summary
r[['params', 'mean_train_score', 'mean_test_score']].head()
# pull out best params, retrain, evaluate
best_alpha = grid_search.best_params_['alpha']
best_max_iter = grid_search.best_params_['max_iter']
print('Best alpha: %s' % best_alpha)
print('Best max_iter: %s' % best_max_iter)
best_model = linear_model.SGDClassifier(
    loss='log', tol=-np.inf, class_weight='balanced', alpha=best_alpha, max_iter=best_max_iter)
best_model.fit(x_train, y_train)
evaluate(best_model, x_train, y_train, x_test, y_test)
"""
Task: Beat the baseline 67.24% test accuracy
"""
def get_feature_set(data, wanted_features):
    return data.loc[:, [col in wanted_features for col in data.columns]]
param_grid = {
    'alpha': [0.0001, 1.],
    'max_iter': [10, 10000]
}
wanted_features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist_true', 'typea',
                   'famhist_false', 'obesity', 'alcohol', 'age', 'chd']  # must have chd - DON'T REMOVE
# this may take a few seconds to load
my_data = get_feature_set(data, wanted_features)  # feature selection
x_train, y_train, x_test, y_test = split(my_data) # splits
grid_search = model_selection.GridSearchCV(       # perform grid search
    estimator=linear_model.SGDClassifier(loss='log', tol=-np.inf, class_weight='balanced'),
    param_grid=param_grid,
    cv=10,
    return_train_score=True)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
best_model = linear_model.SGDClassifier(loss='log', tol=-np.inf, class_weight='balanced',
                                        **grid_search.best_params_)
best_model.fit(x_train, y_train)
evaluate(best_model, x_train, y_train, x_test, y_test)