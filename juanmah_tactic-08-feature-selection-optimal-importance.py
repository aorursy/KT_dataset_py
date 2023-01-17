import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV

from sklearn.utils.multiclass import unique_labels



import time

import pickle



import matplotlib.pyplot as plt



from tqdm import tqdm_notebook



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
from lwoku import get_accuracy#, add_features

accuracy_all_features = get_accuracy(lg_clf, X_train, y_train)

print('Accuracy: {:.4f} %'.format(accuracy_all_features * 100))
model_fitted = lg_clf.fit(X_train, y_train)

    

importances = pd.DataFrame({'Features': X_train.columns, 

                            'Importances': model_fitted.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18, 6))

plt.bar(importances['Features'], importances['Importances'])

plt.xticks(rotation='vertical')

plt.show()
results = pd.DataFrame()

features = []

features_positive = []

features_negative = []

accuracy_last = 0



for feature in tqdm_notebook(importances['Features']):

    features.append(feature)

    accuracy = get_accuracy(lg_clf, X_train[features], y_train)

    if accuracy > accuracy_last:

        features_positive.append(feature)

    else:

        features_negative.append(feature)

    accuracy_last = accuracy

    print('{}: {:.4f} %'.format(feature, accuracy * 100))

    results = results.append({

        'Accuracy': accuracy,

        'Feature': feature,        

        'Features': features.copy(),

        'Features positive': features_positive.copy()

    }, ignore_index = True)



results.to_csv('results.csv', index = True)

results
iteration = results['Accuracy'].idxmax()

features_largest_accuracy = results['Features'][iteration]

print('Best result in iteration #{}'.format(iteration))

print('{} features: {}'.format(len(features_largest_accuracy), features_largest_accuracy))

print('{} features positive: {}'.format(len(features_positive), features_positive))

print('{} features negative: {}'.format(len(features_negative), features_negative))
print('N# features in largest accuracy: {} and in positive: {}'.format(len(features_largest_accuracy), len(features_positive)))



print(list(set(features_largest_accuracy) - set(features_positive)))



print(list(set(features_positive) - set(features_largest_accuracy)))
def submit(features, name):

    

    print(name)

    accuracy = get_accuracy(lg_clf, X_train[features], y_train)

    print('Accuracy: {:.2f} %'.format(accuracy * 100))

    print('Improvement: {:.2f} %\n'.format((accuracy - accuracy_all_features)* 100))



    # Predict

    lg_clf.fit(X_train[features], y_train)

    y_test_pred = pd.Series(lg_clf.predict(X_test[features]))



    # Submit

    output = pd.DataFrame({'ID': X_test.index,

                           'Cover_Type': y_test_pred})



    output.to_csv('submission_' + name + '.csv', index=False)
submit(features_largest_accuracy, 'largest_accuracy')



# submit(results['Features'][iteration - 1], 'largest_accuracy_m1')



# submit(results['Features'][iteration + 1], 'largest_accuracy_p1')



submit(features_positive, 'positive')



# submit(features_positive[:-1], 'positive_m1')



# submit(features_positive + [features_negative[0]], 'positive_p1')