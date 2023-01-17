# packages we will be using
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection
import numpy as np
import pandas as pd
num_hours_studied = np.array([1, 3, 3, 4, 5, 6, 7, 7, 8, 8, 10])
exam_score = np.array([18, 26, 31, 40, 55, 62, 71, 70, 75, 85, 97])
plt.scatter(num_hours_studied, exam_score)
plt.xlabel('num_hours_studied')
plt.ylabel('exam_score')
plt.show()
# Fit the model
exam_model = linear_model.LinearRegression(normalize=True)
x = np.expand_dims(num_hours_studied, 1)
y = exam_score
exam_model.fit(x, y)
a = exam_model.coef_
b = exam_model.intercept_
print(exam_model.coef_)
print(exam_model.intercept_)
# Visualize the results
plt.scatter(num_hours_studied, exam_score)
x = np.linspace(0, 10)
y = a*x + b
plt.plot(x, y, 'r')
plt.xlabel('num_hours_studied')
plt.ylabel('exam_score')
plt.show()
"""
Task: Load the data with pandas
"""
file_name = '../input/prostate.csv'
data = pd.read_csv(file_name, sep='\t', index_col=0)
assert len(data.columns) == 10
assert len(data) == 97
for column_name in ['lcavol', 'lweight', 'age', 'lbph', 'svi', 
                    'lcp', 'gleason', 'pgg45', 'lpsa', 'train']:
    assert column_name in data.columns
print('Success!')
data.head()
# function to help us plot
def scatter(_data, x_name):
    plt.scatter(_data[x_name], _data['lpsa'])
    plt.xlabel(x_name)
    plt.ylabel('lpsa')
    plt.show()

scatter(data, 'pgg45')
"""
Task: Explore the relationships between the response and the other features
"""

"""
Task: Split the data into train and test
"""
train = data[data['train'] == 'T'].drop(['train'], axis=1)
test = data[data['train'] == 'F'].drop(['train'], axis=1)
assert len(train) == 67
assert len(test) == 30
assert 'train' not in train.columns
assert 'train' not in test.columns
assert len(train.columns) == 9
assert len(test.columns) == 9
print('Success!')
train.columns != 'lpsa'
x_train = train.loc[:, train.columns != 'lpsa']
y_train = train['lpsa']
x_test = test.loc[:, test.columns != 'lpsa']
y_test = test['lpsa']
assert len(x_train.columns) == 8
assert len(x_test.columns) == 8
assert len(y_train) == 67
assert len(y_test) == 30
print('Success!')
model = linear_model.LinearRegression(normalize=True)
model.fit(x_train, y_train)
train_pred = model.predict(x_train)
mse_train = metrics.mean_squared_error(y_train, train_pred)
print(mse_train)
np.sqrt(mse_train)
print('lpsa min: %s' % data['lpsa'].min())
print('lpsa max: %s' % data['lpsa'].max())
# get predictions on test set
test_pred = model.predict(x_test)
test_mse = metrics.mean_squared_error(y_test, test_pred)
print('test MSE: %s' % test_mse)
print('test RMSE: %s' % np.sqrt(test_mse))
# this cell tells you what all the feature names are for convenience
x_train.columns
# you can use this cell to visualize here for convenience
scatter(data, 'lcavol')
wanted_features = ['lcavol', 'lweight']
select_features = [column in wanted_features for column in data.columns]
x_train_fs = x_train.loc[:, select_features]
x_train_fs.head()
model_fs = linear_model.LinearRegression(normalize=True)
model_fs.fit(x_train_fs, y_train)
train_preds_fs = model_fs.predict(x_train_fs)
metrics.mean_squared_error(y_train, train_preds_fs)
x_test_fs = x_test.loc[:, select_features]
test_preds_fs = model_fs.predict(x_test_fs)
metrics.mean_squared_error(y_test, test_preds_fs)
def scorer(model, X, y):
    preds = model.predict(X)
    return metrics.mean_squared_error(y, preds)
alphas = np.linspace(start=0, stop=0.5, num=11)
alphas
"""
Task: Perform 10-fold cross validation on all values of alpha and save the mses.
"""
mses = []
for alpha in alphas:
    ridge = linear_model.Ridge(alpha=alpha, normalize=True)
    mse = model_selection.cross_val_score(ridge, x_train, y_train, cv=10, scoring=scorer)
    mses.append(mse.mean())
assert len(mses) == 11             # must have the same number of scores as we have alpha values
assert isinstance(mses[0], float)  # i.e. not an array, i.e. mean() has been called
print('Success!')
plt.plot(alphas, mses)
plt.xlabel('alpha')
plt.ylabel('mse')
plt.show()
best_alpha = alphas[np.argmin(mses)]
best_alpha
ridge = linear_model.Ridge(alpha=best_alpha, normalize=True)
ridge.fit(x_train, y_train)
train_preds = ridge.predict(x_train)
test_preds = ridge.predict(x_test)
train_mse = metrics.mean_squared_error(y_train, train_preds)
test_mse = metrics.mean_squared_error(y_test, test_preds)
print('Train MSE: %s' % train_mse)
print('Test MSE: %s' % test_mse)
for i in range(0, len(train.columns) - 1):
    print('Coefficient for %s:%s\t%s' %
          (train.columns[i], 
           '\t' if len(train.columns[i]) < 7 else '',
           ridge.coef_[i]))
"""
Task: See if you can get a lower test MSE with ridge regression and a subset of features
"""
# step 1: feature selection
wanted_features = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 
                   'lcp', 'gleason', 'pgg45']  # CHANGE THIS!!!
select_features = [column in wanted_features for column in data.columns]
x_train_fs = x_train.loc[:, select_features]
# first fit that without ridge regression and check test set MSE
no_rr_model = linear_model.LinearRegression(normalize=True)
no_rr_model.fit(x_train_fs, y_train)
x_test_fs = x_test.loc[:, select_features]
no_rr_test_preds = no_rr_model.predict(x_test_fs)
metrics.mean_squared_error(y_test, no_rr_test_preds)
# now add ridge regression - first find the best alpha
mses = []
alphas = np.linspace(start=0, stop=0.5, num=11)  # YOU CAN ALSO CHANGE THIS!
for alpha in alphas:
    ridge = linear_model.Ridge(alpha=alpha, normalize=True)
    mse = model_selection.cross_val_score(ridge, x_train_fs, y_train, cv=10, scoring=scorer)
    mses.append(mse.mean())
best_alpha = alphas[np.argmin(mses)]
print(best_alpha)
print(min(mses))
# now train with ridge regression and evaluate on test set
ridge = linear_model.Ridge(alpha=best_alpha, normalize=True)
ridge.fit(x_train_fs, y_train)
test_preds = ridge.predict(x_test_fs)
metrics.mean_squared_error(y_test, test_preds)


