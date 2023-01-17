import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



try:

    train = pd.read_csv('/kaggle/input/learn-together/train.csv')

    print("The dataset has {} observations with {} features".format(*train.shape))

except:

    print("Error when reading the file")
train.shape
train['Cover_Type'].value_counts()
train.drop('Id', axis=1, inplace=True)
import pandas_profiling

train.profile_report()
train.drop(['Soil_Type15', 'Soil_Type17'], axis=1, inplace=True)
#separate target and features

y = train.Cover_Type

train.drop(['Cover_Type'], axis=1, inplace=True)
# split into train and test datasets

X_train, X_valid, y_train, y_valid = train_test_split(train, y, train_size=0.8, test_size=0.2, random_state=1)



# Shape of training data (num_rows, num_columns)

print(X_train.shape)

plt.figure(figsize=(16,10))

sns.heatmap(pd.concat([y, train], axis=1).corr(), cmap="coolwarm", linecolor="white", linewidth=1)
from sklearn import ensemble

clf = ensemble.RandomForestClassifier()

print(clf.get_params())
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

rf_random.fit(X_train, y_train)
rf_random.best_params_
from sklearn.metrics import accuracy_score



def evaluate(model, X, y):

    predictions = model.predict(X)

    errors = abs(y - predictions)

    acc_score = accuracy_score(y, predictions)

    print('Model Performance')

    print('Average Error: {:0.3f}.'.format(np.mean(errors)))

    print('Accuracy = {:0.3f}%.'.format(acc_score))    

    return acc_score



default_model = ensemble.RandomForestClassifier(n_estimators = 10, random_state = 1)

default_model.fit(X_train, y_train)

default_accuracy = evaluate(default_model, X_valid, y_valid)



best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, X_valid, y_valid)



print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - default_accuracy) / default_accuracy))

from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

params = {

    'bootstrap': [False],

    'max_depth': [15, 30],

    'max_features': ['auto'],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [2, 4],

    'n_estimators': [1000, 1400]

}



gsv = GridSearchCV(clf, params, cv=4, n_jobs=-1, scoring='accuracy')

gsv.fit(X_train, y_train)

gsv.best_estimator_.feature_importances_



print(gsv.best_params_)

gsv_accuracy = evaluate(gsv.best_estimator_, X_valid, y_valid)

print('Improvement of {:0.2f}%.'.format( 100 * (gsv_accuracy - default_accuracy) / default_accuracy))

# def feature_importances(clf, X, y, figsize=(18, 6)):

#     clf = clf.fit(X, y)

    

#     importances = pd.DataFrame({'Features': X.columns, 

#                                 'Importances': clf.feature_importances_})

    

#     importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



#     fig = plt.figure(figsize=figsize)

#     sns.barplot(x='Features', y='Importance', data=importances)

#     plt.xticks(rotation='vertical')

#     plt.show()

#     return importances

    

# importances = feature_importances(gsv, X_train, y_train)    
print(classification_report(y_train, gsv.best_estimator_.predict(X_train)))



print(classification_report(y_valid, gsv.best_estimator_.predict(X_valid)))
from sklearn.metrics import accuracy_score

acc_score_train = accuracy_score(y_train, gsv.best_estimator_.predict(X_train))

acc_score_valid = accuracy_score(y_valid, gsv.best_estimator_.predict(X_valid))

print('Accuracy score (training data): {}'.format(acc_score_train))

print('Accuracy score (validation data): {}'.format(acc_score_valid))
try:

    test = pd.read_csv('/kaggle/input/learn-together/test.csv')

    print("The test dataset has {} observations with {} features".format(*test.shape))

except:

    print("Error when reading the file")
test.head()
# exclude the removed features (Id, Soil_Type15 and Soil_Type7)

used_features = train.columns

predictions =  gsv.best_estimator_.predict(test[used_features])#[:, 1]  

#Create a  DataFrame with predictions

submission = pd.DataFrame({'Id': test['Id'].tolist(), 'Cover_Type':predictions.tolist()})

filename = 'My submission Forest Types.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)