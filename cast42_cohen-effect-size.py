# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from matplotlib import pyplot as plt
import seaborn as sns

seed = 1234
random.seed(seed)
np.random.seed(seed)
import os
input_files = os.listdir("../input")
print(input_files)
# for filename in input_files:
#     locals()[filename.rstrip('.csv')] = pd.read_csv(f'../input/{filename}')
application_train = pd.read_csv(f'../input/application_train.csv')
application_test  = pd.read_csv(f'../input/application_test.csv')
sample_submission = pd.read_csv(f'../input/sample_submission.csv')
# categoricalize
categorical_columns = ['NAME_CONTRACT_TYPE',
                       'CODE_GENDER',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE',
                       'FONDKAPREMONT_MODE',
                       'HOUSETYPE_MODE',
                       'WALLSMATERIAL_MODE',
                       'EMERGENCYSTATE_MODE']

for column in categorical_columns:
    application_train[column] = application_train[column].astype('category')
    application_test[column] = application_test[column].astype('category')
input_columns = application_train.columns
input_columns = input_columns[input_columns != 'TARGET']
target_column = 'TARGET'

X = application_train[input_columns]
y = application_train[target_column]
def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
    """
    group1 = X[y==0]
    group2 = X[y==1]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1 = group1.shape[0]
    n2 = group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d

from sklearn.utils import shuffle

def p_value_effect(X, y, nr_iters=1000):
    """Calculates the Cohen effect size of each feature and obtains a p-value by bootstrapping.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        p_value : array, shape = [n_features,]
            The set of p-values.
    """
    
    actual = cohen_effect_size(X, y)
    results = np.zeros(actual.shape[0])
    actual_es_abs = actual.abs()
    y_shuffled = y.copy()
    for i in range(nr_iters):
        y_shuffled = shuffle(y_shuffled)
        results = results + (cohen_effect_size(X, y_shuffled.values).abs() >= actual_es_abs)
    p_values = results/nr_iters
    return pd.DataFrame({'cohen_effect_size':actual, 'p_value':p_values}, index=actual.index)
X_effect = X.select_dtypes(include=[np.number])
X_effect.shape
y.value_counts()
effect_sizes = cohen_effect_size(X_effect, y)
effect_sizes.head()
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).index).dropna().head(20)
%time effect_sizes = p_value_effect(X_effect, y, 1000)
effect_sizes.reindex(effect_sizes.cohen_effect_size.abs().sort_values(ascending=False).index).dropna()
significant_effects = effect_sizes[(effect_sizes.p_value <= 0.05)]
significant_effects.reindex(significant_effects.cohen_effect_size.abs().sort_values(ascending=False).index).dropna()
list(significant_effects.index.values) + categorical_columns
selected_features = list(significant_effects.index.values) + categorical_columns
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, random_state=seed)

lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_eval = lgb.Dataset(data=X_test, label=y_test)

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'num_iteration': 200,
        'verbose': 0
}

# train
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)
lgb.plot_importance(gbm, figsize=(20, 20))
set(application_test.columns) - set(selected_features)
X_test = application_test[selected_features]
X_test.shape
X_train.shape
set(selected_features) - set(X_test.columns)
# import time
# pred = gbm.predict(X_test)
# submission = sample_submission
# submission.TARGET = pred
# submission.to_csv(f"{time.strftime('%Y_%m_%d_%d_%M')}_submission.csv", index=None)
