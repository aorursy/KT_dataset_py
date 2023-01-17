import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

warnings.filterwarnings("ignore")
# print(os.listdir("../input"))
# print(open('../input/data_description.txt').read())
# Some functions I carry over my notebooks. Some of them won't be used in this exercise.
def show_correlation(df, feature_name, target_name, plot=True, plot_kind='bar'):
    '''
    Shows correlation (not necessarily statistic meaning) between a feature and its target
    '''
    try:
        print('corr', df[feature_name].corr(df[target_name]))
    except:
        pass

    gb = df[[feature_name, target_name]].groupby(feature_name).agg('mean')
    print(gb)
    
    if plot:
        gb.plot(kind=plot_kind)


def onehot(df, column_name):
    '''
    Transforms a particular categorical field in "onehot"
    '''
    dummies_df = pd.get_dummies(df[column_name], prefix=f'_OneHot{column_name}')
    new_column_names = list(dummies_df.columns)
    return (
        pd.concat([df, dummies_df], axis=1),
        new_column_names
    )


def show_columns_with_null(df, threshold=0):
    '''
    Prints all columns in a dataframe that contains NULL data
    Adapted from https://stackoverflow.com/a/53673717/1836321
    '''
    for column in df:
        null_qty = df[column].isnull().sum()
        if null_qty > threshold:
            print(f'{column} has {null_qty} null values')
            

def remove_outliers(dataframe, column, threshold=1.5):
    '''
    Given a column of interest, removes outliers from dataframe. 
    Keeps values between (mean - 1.5std, mean + 1.5std range). Threshold can be changed. 
    '''
    mean = dataframe[column].mean()
    std = dataframe[column].std()
    lower_threshold = mean - threshold * std
    upper_threshold = mean + threshold * std
    return dataframe[dataframe[column].between(lower_threshold, upper_threshold)] 


def get_best_estimator(X_train, y_train, options, scoring, greater_is_better=True):
    '''
    Returns the best estimator that fits this model
    Expects a datastructure like [(estimator_class, param_grid), ...]
    '''
    best_score = (1 if greater_is_better else -1) * math.inf
    best_estimator = None
    best_params = None

    for estimator, tuned_parameters in options:
        if not tuned_parameters:
            default_params = estimator().get_params().copy()
            x1 = list(default_params.items())[0]
            tuned_parameters = {x1[0]: [x1[1]]}

        clf = GridSearchCV(estimator(), tuned_parameters, scoring=scoring)
        clf.fit(X_train, y_train)

        new_best_score = any([
            (greater_is_better and clf.best_score_ < best_score),
            (not(greater_is_better) and clf.best_score_ > best_score),
        ])

        if new_best_score:
            best_score = clf.best_score_
            best_estimator = estimator
            best_params = clf.best_params_

    print(
        f'Best estimator: {best_estimator.__name__} with {best_params} '
        f'(best score: {np.sqrt(np.abs(best_score)):.2f})'
    )
    return best_estimator(**best_params)


def evaluate_features_accuracy(X_train, y_train, X_test, y_test, estimator, scoring):
    ''' Prints score results for an estimator. '''
    pipeline = make_pipeline(
        MinMaxScaler(),
        PolynomialFeatures(),
        estimator,
    )
    pipeline.fit(X_train, y_train)

    scores = cross_val_score(pipeline, X_test, y_test, scoring=scoring)
    
    if scoring == 'neg_mean_squared_error':
        scores = [np.sqrt(abs(x)) for x in scores]  # Convert negative MSE to positive RMSE 
    
    print(f'Score: ~{int(np.mean(scores))} (scores: {[int(x) for x in scores]})')
df = pd.read_csv('../input/train.csv')
target = 'SalePrice'

print(f'DF shape before cleaning Nan is {df.shape}')

# show_columns_with_null(df, threshold=0)
del df['PoolQC']
del df['FireplaceQu']
del df['Fence']
del df['Alley']
del df['MiscFeature']
del df['LotFrontage']
del df['Utilities']

df.dropna(inplace=True)
print(f'DF shape after cleaning Nan is {df.shape}')
df1 = df.copy()

# Most of the fields treated below are just so we can play around with data
# Most of them are not being used in our final model features.
df1, mszoning_onehot_columns = onehot(df1, 'MSZoning')
df1, lotshape_onehot_columns = onehot(df1, 'LotShape')
df1, landcontour_onehot_columns = onehot(df1, 'LandContour')
df1, lotconfig_onehot_columns = onehot(df1, 'LotConfig')
df1, landslope_onehot_columns = onehot(df1, 'LandSlope')
df1, neighborhood_onehot_columns = onehot(df1, 'Neighborhood')
df1, foundation_onehot_columns = onehot(df1, 'Foundation')
df1, salecondition_onehot_columns = onehot(df1, 'SaleCondition')
df1, saletype_onehot_columns = onehot(df1, 'SaleType')

rank_mapping = {
    'Ex': 10,
    'Gd': 8,
    'TA': 6,
    'Fa': 4,
    'Po': 2,
}
df1['ExterQual'] = df1['ExterQual'].map(rank_mapping)
df1['ExterCond'] = df1['ExterCond'].map(rank_mapping)
df1['KitchenQual'] = df1['KitchenQual'].map(rank_mapping)

df1['TotalArea'] = df1['1stFlrSF'] + df1['2ndFlrSF']

df1 = remove_outliers(df1, 'SalePrice')
df1 = remove_outliers(df1, 'LotArea')
df1 = remove_outliers(df1, 'TotalArea')
def evaluate_estimator_for_features(candidate_features):
    print('Evaluate estimator score based on features {}'.format(candidate_features))
    X = df1[candidate_features]
    y = df1[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    estimator = get_best_estimator(
        X_train, y_train, options, scoring='neg_mean_squared_error', greater_is_better=False
    )
    evaluate_features_accuracy(
        X_train, y_train, X_test, y_test, estimator=estimator, scoring='neg_mean_squared_error'
    )
    print('-' * 10)


# Options taken from https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
alpha_candidates = [(1/10**x) for x in range(0, 3)]
penalty_candidates = ['l1', 'l2', 'elasticnet']
options = [
    (linear_model.SGDRegressor, [{'penalty': penalty_candidates, 'alpha': alpha_candidates}]),
    (linear_model.Lasso, [{'alpha': alpha_candidates}]),
    (linear_model.ElasticNet, [{'alpha': alpha_candidates}]),
    (linear_model.LogisticRegression, None),
    (linear_model.LinearRegression, None),
]


# Best features I found for the final model.
candidate_features = [
    'LotArea',
    'YearBuilt',
    'OverallQual',
    'OverallCond',
    'ExterQual',
    'ExterCond',
    'KitchenQual',
    'TotalArea',
]
evaluate_estimator_for_features(candidate_features)


# Code below is here just for fun/demonstration sake - features chosen are not good to final model.
candidate_features = ['Id']
evaluate_estimator_for_features(candidate_features)

candidate_features = mszoning_onehot_columns
evaluate_estimator_for_features(candidate_features)

candidate_features = ['MoSold', 'YrSold']
evaluate_estimator_for_features(candidate_features)

