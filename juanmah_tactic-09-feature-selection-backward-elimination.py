import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np

import pandas as pd



from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_predict, cross_val_score

from tqdm import tqdm_notebook

import pickle



from lwoku import get_accuracy, add_features
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



# Define the dependent variable 

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis='columns')
# Add new features

print('     Number of features (before adding): {}'.format(len(X_train.columns)))

X_train = add_features(X_train)

X_test = add_features(X_test)

print('     Number of features (after adding): {}'.format(len(X_train.columns)))
print('  -- Drop features')

print('    -- Drop columns with few values (or almost)')

features_to_drop = ['Soil_Type7', 'Soil_Type8', 'Soil_Type15', 'Soil_Type27', 'Soil_Type37']

X_train.drop(features_to_drop, axis='columns', inplace=True)

X_test.drop(features_to_drop, axis='columns', inplace=True)

print('    -- Drop columns with low importance')

features_to_drop = ['Soil_Type5', 'Slope', 'Soil_Type39', 'Soil_Type18', 'Soil_Type20', 'Soil_Type35', 'Soil_Type11',

                    'Soil_Type31']

X_train.drop(features_to_drop, axis='columns', inplace=True)

X_test.drop(features_to_drop, axis='columns', inplace=True)

print('     Number of features (after drop): {}'.format(len(X_train.columns)))
with open('../input/tactic-03-hyperparameter-optimization-lightgbm/clf.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lg_clf = clf.best_estimator_

lg_clf
lg_clf_fast = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

               importance_type='split', learning_rate=0.7, max_depth=21,

               min_child_samples=5, min_child_weight=1e-15, min_split_gain=0.0,

               n_estimators=150, n_jobs=-1, num_leaves=30, objective=None,

               random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

               subsample=1.0, subsample_for_bin=987, subsample_freq=0,

               verbosity=1)

lg_clf_fast
%%time

## Get accuracy for the original data without feature engineering

accuracy = get_accuracy(lg_clf_fast, X_train, y_train)

print('Accuracy without feature engineering: {:.2f}\u00A0%'.format(accuracy * 100))
# noinspection PyPep8Naming

def accuracy_table_by_reducing():

    """

    This function calculates the accuracy of the current set of features, and put it in 'Base' column.

    Then it calculates the accuracy if only one feature is dropped.

    The feature set with the greatest accuracy, means that a new set of features without this feature, scores more.

    Then this feature is dropped.

    Finally, the functions continues with this new set until there are only two values left. 

    :return: A table with the base accuracy and the accuracy of the dropped feature for each iteration

    """

    table = pd.DataFrame(columns=['Base'])

    X_features = X_train.copy()



    progress_bar = tqdm_notebook(total=(len(X_features.columns) + 1) * len(X_features.columns) / 2)

    while len(X_features.columns) > 1:

        print(len(X_features.columns))

        row = {'Base': get_accuracy(lg_clf_fast, X_features, y_train)}

        for feature in X_features.columns:

            row[feature] = get_accuracy(lg_clf_fast, X_features.drop(feature, axis='columns'), y_train)

        progress_bar.update(len(X_features.columns))

#         print(row)

        table = table.append(row, ignore_index=True)

        feature_to_drop = table.iloc[-1][1:].idxmax()

        print(feature_to_drop)

        X_features.drop(feature_to_drop, axis='columns', inplace=True)

    progress_bar.close()

    return table





# Get the table for reducing

reducing_table = accuracy_table_by_reducing()
reducing_table.to_csv('reducing_table.csv', index=False)
# Get the largest accuracy

reducing_table_accuracy = reducing_table['Base'].max()

print('Maximum accuracy: {:.2f}\u00A0%'.format(reducing_table_accuracy * 100))



# Get the features set with largest accuracy

reducing_table_best_row = reducing_table.iloc[reducing_table['Base'].idxmax()]

reducing_table_best_row.dropna(inplace=True)

reducing_table_best_row.drop('Base', inplace=True)

reducing_table_features = reducing_table_best_row.axes[0].tolist()

print('For this set of features: {}'.format(reducing_table_features))
# Predict with test data

lg_clf.fit(X_train[reducing_table_features], y_train)

y_test_pred = pd.Series(lg_clf.predict(X_test[reducing_table_features]))



# Submit

output = pd.DataFrame({'ID': X_test.index, 'Cover_Type': y_test_pred})



output.to_csv('submission.csv', index=False)