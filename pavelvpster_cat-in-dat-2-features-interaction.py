import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')
train.head(3).T
def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary





summary(train)
train['missing_count'] = train.isnull().sum(axis=1)

test['missing_count'] = test.isnull().sum(axis=1)
missing_number = -99999

missing_string = 'MISSING_STRING'
numerical_features = [

    'bin_0', 'bin_1', 'bin_2',

    'ord_0',

    'day', 'month'

]



string_features = [

    'bin_3', 'bin_4',

    'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
def impute(train, test, columns, value):

    for column in columns:

        train[column] = train[column].fillna(value)

        test[column] = test[column].fillna(value)
impute(train, test, numerical_features, missing_number)

impute(train, test, string_features, missing_string)
train['ord_5_1'] = train['ord_5'].str[0]

train['ord_5_2'] = train['ord_5'].str[1]



train.loc[train['ord_5'] == missing_string, 'ord_5_1'] = missing_string

train.loc[train['ord_5'] == missing_string, 'ord_5_2'] = missing_string



train = train.drop('ord_5', axis=1)





test['ord_5_1'] = test['ord_5'].str[0]

test['ord_5_2'] = test['ord_5'].str[1]



test.loc[test['ord_5'] == missing_string, 'ord_5_1'] = missing_string

test.loc[test['ord_5'] == missing_string, 'ord_5_2'] = missing_string



test = test.drop('ord_5', axis=1)
simple_features = [

    'missing_count'

]



ohe_features = [

    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

    'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

    'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5_1', 'ord_5_2',

    'day', 'month'

]



target_features = [

    'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

]
y_train = train['target'].copy()

x_train = train.drop('target', axis=1)

del train



x_test = test.copy()

del test
from sklearn.preprocessing import StandardScaler





scaler = StandardScaler()

simple_x_train = scaler.fit_transform(x_train[simple_features])

simple_x_test = scaler.transform(x_test[simple_features])
from sklearn.preprocessing import OneHotEncoder





ohe = OneHotEncoder(dtype='uint16', handle_unknown='ignore')

ohe_x_train = ohe.fit_transform(x_train[ohe_features])

ohe_x_test = ohe.transform(x_test[ohe_features])
from category_encoders import TargetEncoder

from sklearn.model_selection import StratifiedKFold
def transform(transformer, x_train, y_train, cv):

    oof = pd.DataFrame(index=x_train.index, columns=x_train.columns)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train.loc[train_idx]

        y_train_train = y_train.loc[train_idx]

        x_train_valid = x_train.loc[valid_idx]

        transformer.fit(x_train_train, y_train_train)

        oof_part = transformer.transform(x_train_valid)

        oof.loc[valid_idx] = oof_part

    return oof
target = TargetEncoder(drop_invariant=True, smoothing=0.2)



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

target_x_train = transform(target, x_train[target_features], y_train, cv).astype('float')



target.fit(x_train[target_features], y_train)

target_x_test = target.transform(x_test[target_features]).astype('float')
import scipy





final_x_train = scipy.sparse.hstack([simple_x_train, ohe_x_train, target_x_train]).tocsr()

final_x_test = scipy.sparse.hstack([simple_x_test, ohe_x_test, target_x_test]).tocsr()
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
def test(x_train, y_train):

    logit = LogisticRegression(C=0.54321, solver='lbfgs', max_iter=5000)

    scores = []

    cv = KFold(n_splits=5, random_state=42)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train[train_idx]

        y_train_train = y_train[train_idx]

        x_train_valid = x_train[valid_idx]

        y_train_valid = y_train[valid_idx]

        logit.fit(x_train_train, y_train_train)

        y_pred = logit.predict_proba(x_train_valid)[:, 1]

        score = roc_auc_score(y_train_valid, y_pred)

        print('Fold score:', score)

        scores.append(score)

    score = np.mean(scores)

    print('Average score:', score)

    return score
baseline_score = test(final_x_train, y_train)

baseline_score
# Modify these parameters



features_to_test = ohe_features + target_features



offset = 0



run_time_limit = 120 * 60 # sec
import itertools





feature_combinations = list(itertools.combinations(features_to_test, 2))

print('Total number of combinations:', len(feature_combinations))



feature_combinations = feature_combinations[offset:]

print('Number of combinations to test:', len(feature_combinations))



# for features in feature_combinations:

#     print(features)
import time





start_time = time.time()



new_features = {}



for features in feature_combinations:

    new_feature = features[0] + '__' + features[1]

    print('Test feature:', new_feature, '/ interaction of', features[0], 'and', features[1])

    

    temp = pd.DataFrame(index=x_train.index)

    temp[new_feature] = x_train[features[0]].astype(str) + '_' + x_train[features[1]].astype(str)

    ohe = OneHotEncoder(dtype='uint16', handle_unknown='ignore')

    encoded_temp = ohe.fit_transform(temp)

    

    score = test(scipy.sparse.hstack([final_x_train, encoded_temp]).tocsr(), y_train)

    print('Score =', score)

    

    new_features[new_feature] = score

    

    # limit execution time

    run_time = time.time() - start_time

    if run_time > run_time_limit:

        break
new_features_df = pd.DataFrame.from_dict(new_features, orient='index', columns=['score']).sub(baseline_score)

new_features_df
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(12, 12))

p = sns.barplot(x=new_features_df.index, y='score', data=new_features_df, color='gray')



# Rotate labels

for x in p.get_xticklabels():

    x.set_rotation(90)



plt.show()
next_run_offset = offset + len(new_features_df)

next_run_offset
new_features_df[new_features_df['score'] > 0].index.tolist()