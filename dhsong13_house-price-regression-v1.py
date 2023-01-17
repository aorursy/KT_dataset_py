# Python Standard Library

import os



# Numerical Data

import numpy as np



# DataFrame

import pandas as pd



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Machine Learning

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from sklearn.feature_selection import RFECV



from sklearn.metrics import mean_squared_error



from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import ExtraTreeRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.neighbors import KNeighborsRegressor



from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import VotingRegressor
pd.options.display.max_columns = 499

pd.options.display.max_rows = 499

pd.options.mode.chained_assignment = None
%matplotlib inline
train_raw = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', encoding='utf-8')

test_raw = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', encoding='utf-8')
train_raw.shape
train_raw.head()
train_raw.info()
train_raw.describe(include='all')
train_raw.isna().sum()[train_raw.isna().sum() > 0]
test_raw.shape
test_raw.head()
test_raw.info()
test_raw.describe(include='all')
test_raw.isna().sum()[test_raw.isna().sum() > 0]
train_cleaned = train_raw.select_dtypes(exclude='object')

test_cleaned = test_raw.select_dtypes(exclude='object')
train_cleaned
test_cleaned
missing_columns = list()

for col in test_cleaned.columns:

    if train_cleaned[col].isna().any() or test_cleaned[col].isna().any():

        missing_columns.append(col)

print(missing_columns)
def show_boxplot(dataset, columns):

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))

    

    for idx, column in enumerate(columns):

        row = idx // 4

        col = idx % 4

        sns.boxplot(dataset[column], orient='v', ax=axes[row][col])

    plt.show()
show_boxplot(train_cleaned, missing_columns)
def show_hist(dataset, columns):

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))

    

    for idx, column in enumerate(columns):

        row = idx // 4

        col = idx % 4

        sns.distplot(dataset[column], ax=axes[row][col], kde_kws={'bw':0.1})

    plt.show()    
show_hist(train_cleaned, missing_columns)
def impute_missing_values(train, test):

    median_columns = ['LotFrontage', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']

    top_columns = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 

                   'GarageYrBlt', 'GarageCars']

    

    imputation_dict = dict()

    for col in median_columns:

        median_value = train[col].median()

        imputation_dict[col] = median_value

        train[col] = train[col].fillna(median_value)

        test[col] = test[col].fillna(median_value)



    top_dict = dict()

    for col in top_columns:

        top_value = train[col].value_counts().index[0]

        imputation_dict[col] = top_value

        train[col] = train[col].fillna(top_value)

        test[col] = test[col].fillna(top_value)

    

    

    return imputation_dict
imputation_dict = impute_missing_values(train_cleaned, test_cleaned)

imputation_dict
def heatmap(dataset):

    plt.figure(figsize=(12, 10))

    sns.heatmap(train_cleaned.corr())

    plt.show()
heatmap(train_cleaned)
x_features = train_cleaned.columns[1:-1]

y_features = train_cleaned.columns[-1:]
x_train = train_cleaned.loc[:, x_features]

x_test = test_cleaned.loc[:, x_features]
y_train = train_cleaned.loc[:, y_features]
y_test_id = test_cleaned.loc[:, 'Id']
x_train, x_val, y_train, y_val = train_test_split(

    x_train, 

    y_train, 

    test_size=0.2,

    random_state=2020

)
def _grid_search(model, parameters, x_train, y_train):

    rgs = GridSearchCV(

        model, parameters, 

        n_jobs=4,

        cv=5, 

        verbose=2, 

        return_train_score=True

    )

    rgs.fit(x_train, y_train)

    return rgs
def svr_grid_search(x_train, y_train, x_val, y_val):

    model = SVR()

    parameters = {

        'kernel': ('linear', 'rbf'),

        'C': (0.5, 1.0),

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

svr_rgs = svr_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def dt_grid_search(x_train, y_train, x_val, y_val):

    model = DecisionTreeRegressor(random_state=2020)

    parameters = {

        'max_depth': (None, 1, 3, 5, 7),

        'min_samples_split': (2, 3, 4, 5),

        'min_samples_leaf': (1, 3, 5)        

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

dt_rgs = dt_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def et_grid_search(x_train, y_train, x_val, y_val):

    model = ExtraTreeRegressor(random_state=2020)

    parameters = {

        'max_depth': (None, 1, 3, 5, 7),

        'min_samples_split': (2, 3, 4, 5),

        'min_samples_leaf': (1, 3, 5)        

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

et_rgs = et_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def knn_grid_search(x_train, y_train, x_val, y_val):

    model = KNeighborsRegressor()

    parameters = {

        'n_neighbors': (1, 3, 5, 7, 9)        

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

knn_rgs = knn_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def mlp_grid_search(x_train, y_train, x_val, y_val):

    model = MLPRegressor()

    parameters = {

        'hidden_layer_sizes': (100, 150),

        'activation': ('relu', 'sigmoid', 'tanh'),

        'learning_rate': ('constant', 'invscaling', 'adaptive'),

        'max_iter': (100, 200)

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

mlp_rgs = mlp_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def ada_grid_search(x_train, y_train, x_val, y_val):

    model = AdaBoostRegressor(random_state=2020)

    parameters = {

        'n_estimators': (100, 300, 500),

        'learning_rate': (0.01, 0.1, 1.0),

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

ada_rgs = ada_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def rf_grid_search(x_train, y_train, x_val, y_val):

    model = RandomForestRegressor(random_state=2020)

    parameters = {

        'n_estimators': (10, 30, 50),

        'max_depth': (None, 3, 5, 7),

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

rf_rgs = rf_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def gd_grid_search(x_train, y_train, x_val, y_val):

    model = GradientBoostingRegressor(random_state=2020)

    parameters = {

        'n_estimators': (100, 300, 500),

        'learning_rate': (0.001, 0.01, 0.1),

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

gd_rgs = gd_grid_search(x_train, y_train_1d, x_val, y_val_1d)
def vt_grid_search(estimators, x_train, y_train, x_val, y_val):

    model = VotingRegressor(estimators=estimators)

    parameters = {

        'weights': (None, [1, 2], [2, 1])

    }



    rgs = _grid_search(model, parameters, x_train, y_train)

    

    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data: {}'.format(rmse))

    

    return rgs    

    
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

estimators = [

    ('rf', rf_rgs.best_estimator_),

    ('gd', gd_rgs.best_estimator_)

]

vt_rgs = vt_grid_search(estimators, x_train, y_train_1d, x_val, y_val_1d)
def _feature_selection(estimator, x_train, y_train, x_val, y_val, x_test):

    selector = RFECV(estimator, step=1, min_features_to_select=30, cv=5)    

    

    x_train_new = selector.fit_transform(x_train, y_train)

    x_val_new = selector.transform(x_val)

    x_test_new = selector.transform(x_test)

    

    return estimator, x_train_new, x_val_new, x_test_new
def gd_feature_selection(x_train, y_train, x_val, y_val, x_test):

    model = GradientBoostingRegressor(random_state=2020)

    parameters = {

        'n_estimators': (100, 300, 500),

        'learning_rate': (0.001, 0.01, 0.1),

    }

    

    rgs = _grid_search(model, parameters, x_train, y_train)



    prediction = rgs.predict(x_val)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data (Before Feture Selection): {}'.format(rmse))



    

    rgs_fs, x_train_new, x_val_new, x_test_new = _feature_selection(

        rgs.best_estimator_, 

        x_train, y_train, 

        x_val, y_val, 

        x_test

    )

    

    rgs_fs.fit(x_train_new, y_train)

    

    prediction = rgs_fs.predict(x_val_new)

    rmse = mean_squared_error(y_val, prediction, squared=False)

    

    print('Mean Square Error on Validation Data (After Feture Selection): {}'.format(rmse))

    

    return rgs_fs, x_train_new, x_val_new, x_test_new
y_train_1d = np.ravel(y_train)

y_val_1d = np.ravel(y_val)

gd_rgs_fs, x_train_new, x_val_new, x_test_new = gd_feature_selection(x_train, y_train_1d, x_val, y_val_1d, x_test)
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = gd_rgs.predict(x_test)

submission
os.makedirs('/kaggle/working/sumbission/', exist_ok=True)

submission.to_csv('/kaggle/working/sumbission/submission-v1.csv', index=False, encoding='utf-8')