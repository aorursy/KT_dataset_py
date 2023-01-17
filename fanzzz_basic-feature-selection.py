# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def feature_selection(feature_matrix, missing_threshold=90, correlation_threshold=0.95):
    """Feature selection for a dataframe."""
    
    feature_matrix = pd.get_dummies(feature_matrix)
    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

    _, idx = np.unique(feature_matrix, axis = 1, return_index = True)
    feature_matrix = feature_matrix.iloc[:, idx]
    n_non_unique_columns = n_features_start - feature_matrix.shape[1]
    print('{}  non-unique valued columns.'.format(n_non_unique_columns))

    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['percent'] = 100 * (missing[0] / feature_matrix.shape[0])
    missing.sort_values('percent', ascending = False, inplace = True)

    # Missing above threshold
    missing_cols = list(missing[missing['percent'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols,
                                                                        missing_threshold))
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_non_unique_columns + n_missing_cols + n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    return feature_matrix
print(os.listdir('../input'))
# print(os.listdir('../input/home-credit-default-risk-feature-tools'))
# time_start = time.time()
# feature_matrix = pd.read_csv('../input/home-credit-default-risk-feature-tools/feature_matrix.csv', low_memory=False)
# feature_matrix.shape

# # Sampling 10% of the original data
# auto_features = feature_matrix[feature_matrix['TARGET'].notnull()].sample(frac = 0.1, random_state = 50)

# import gc
# gc.enable()
# del feature_matrix
# gc.collect()
# for col in train:
#     if train[col].dtype == 'bool':
#         train[col] = train[col].astype(np.uint8)
# auto_features = feature_selection(train, 90, 0.95)
# ### 2091 --> 1045
# auto_features.to_csv('../working/features_automated_sample.csv', index = False)
# fm = pd.read_csv('../input/feature_matrix.csv')
# # One-hot encoding
# fm = pd.get_dummies(fm)
# print(fm.shape)
# # Subset to the columns in the sample
# fm = fm[auto_features.columns]
# print(fm.shape)
# fm.to_csv('../working/features_auto_selected.csv')
# time_end = time.time()
# print(time_end - time_start, 's')
os.listdir('../input/semi-automated-feature-engineering-by-willkoehrsen/')
import pandas as pd
semi_features  = pd.read_csv('../input/semi-automated-feature-engineering-by-willkoehrsen/final_manual_features.csv')
test = semi_features[semi_features['TARGET'].isnull()]
print(test.shape)
time_start = time.time()
semi_features  = pd.read_csv('../input/semi-automated-feature-engineering-by-willkoehrsen/final_manual_features.csv')
semi_features = semi_features[semi_features['TARGET'].notnull()].sample(frac = 0.1, random_state = 50)
print(semi_features.shape)
semi_features = feature_selection(semi_features, 90, 0.95)
### 1458 --> 874
semi_features.to_csv('../working/features_semi_sample.csv', index = False)
fm = pd.read_csv('../input/semi-automated-feature-engineering-by-willkoehrsen/final_manual_features.csv')
# One-hot encoding
fm = pd.get_dummies(fm)
print(fm.shape)
# Subset to the columns in the sample
fm = fm[semi_features.columns]
print(fm.shape)
fm.to_csv('../working/features_semi_selected.csv')
time_end = time.time()
print(time_end - time_start, 's')
os.listdir('../input/manual-feature-engineering-by-willkoehrsen')
time_start = time.time()
manual_features  = pd.read_csv('../input/manual-feature-engineering-by-willkoehrsen/features_manual_domain.csv')
manual_features = manual_features[manual_features['TARGET'].notnull()].sample(frac = 0.1, random_state = 50)

manual_features = feature_selection(manual_features, 90, 0.95)
### 275 --> 232
manual_features.to_csv('../working/features_manual_sample.csv', index = False)
fm = pd.read_csv('../input/manual-feature-engineering-by-willkoehrsen/features_manual_domain.csv')
# One-hot encoding
fm = pd.get_dummies(fm)
print(fm.shape)
# Subset to the columns in the sample
fm = fm[manual_features.columns]
print(fm.shape)
fm.to_csv('../working/features_manual_selected.csv')
time_end = time.time()
print(time_end - time_start, 's')
os.listdir('../input/feature-cleaning-after-understanding-variables')
time_start = time.time()
default_features = pd.read_csv('../input/feature-cleaning-after-understanding-variables/df.csv')
default_features = default_features[default_features['TARGET'].notnull()].sample(frac = 0.1, random_state = 50)

default_features = pd.get_dummies(default_features)
default_features.shape
default_features = feature_selection(default_features, 90, 0.95)
default_features.head()
### 232 --> 231
default_features.to_csv('../working/features_default_sample.csv', index = False)
# train, test = pd.read_csv('../input/application_train.csv'), pd.read_csv('../input/application_test.csv')
# test['TARGET'] = np.nan
# train, test = pd.get_dummies(train).align(pd.get_dummies(test), axis = 1, join = 'inner')
# fm = train.append(test, ignore_index = True)
fm = pd.read_csv('../input/feature-cleaning-after-understanding-variables/df.csv')
fm = pd.get_dummies(fm)
print(fm.shape)
fm = fm[default_features.columns]
fm.to_csv('../working/features_default_selected.csv')
print(fm.shape)
time_end = time.time()
print(time_end - time_start, 's')



