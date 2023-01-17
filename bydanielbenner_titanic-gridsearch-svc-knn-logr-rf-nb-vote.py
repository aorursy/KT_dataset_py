# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
# read files and first glance



data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv')



data_total = data_train.append(data_test, ignore_index=True)

data_train_length = len(data_train.index)



for c in ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Survived']:

    sns.catplot(x=c, kind='count', data=data_train)

    plt.show()



for c in ['Age', 'Fare']:

    data_train[c].hist().set(xlabel=c)

    plt.show()



for c in ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']:

    sns.catplot(x=c, y='Survived', kind='bar', data=data_train)

    plt.show()





for c in ['Age', 'Fare']:

    g = sns.FacetGrid(data_train, col='Survived')

    g.map(plt.hist, c)

    plt.show()





# check correlations



corr = data_train.corr() # correlations between age, pclass, fare



sns.boxplot(x='Pclass', y='Age', hue='Survived', data=data_train)

plt.show()

sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data_train[data_train['Fare'] < 200])

plt.show()

data_total.info()
# impute missing values ========================================



# fill age na with mean of age for pclass

age_mean_by_pclass = data_total[['Pclass', 'Age']].groupby('Pclass').mean()

data_total['Age'] = data_total.apply(

    lambda row: age_mean_by_pclass.loc[row['Pclass'], 'Age'] if np.isnan(row['Age']) else row['Age'], axis=1)



# fill fare na with median of fare for pclass

fare_median_by_pclass = data_total[['Pclass', 'Fare']].groupby('Pclass').median()

data_total['Fare'] = data_total.apply(

    lambda row: fare_median_by_pclass.loc[row['Pclass'], 'Fare'] if np.isnan(row['Fare']) else row['Fare'], axis=1)



# fill embarked na with mode 

data_total['Embarked'].fillna(data_total['Embarked'].mode()[0], inplace=True)



data_total.info()
# create and drop features ========================================



# create feature for people who are alone or in a big group

data_total['isAlone'] = ((data_total['SibSp'] + data_total['Parch']) == 0) * 1

data_total['bigGroup'] = ((data_total['SibSp'] + data_total['Parch']) > 3) * 1



# create feature for title

data_total['Title'] = data_total['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

data_total['Title'] = data_total['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

data_total['Title'] = data_total['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')



# convert age to bins

age_bin_count = 10

data_total['Age_Bin'] = pd.cut(data_total['Age'], age_bin_count, labels=list(range(1,age_bin_count+1)))



# convert fare to bins

fare_bin_count = 5

data_total['Fare_Bin'] = pd.qcut(data_total['Fare'], fare_bin_count, labels=list(range(1,fare_bin_count+1)))



# onehotencode sex, embarked, title

enc = OneHotEncoder().fit(data_total[['Sex', 'Embarked', 'Title']])

enc_features_arr = enc.transform(data_total[['Sex', 'Embarked', 'Title']]).toarray()

enc_features_labels = [label for cat in enc.categories_ for label in cat] # flatten features_arr like .ravel()

enc_features_df = pd.DataFrame(enc_features_arr, columns=enc_features_labels).drop(columns=['male', 'S', 'Other'])

data_total = data_total.join(enc_features_df)



# drop useless features

data_total = data_total.drop(columns=['PassengerId', 'Name', 'Cabin', 'Ticket', 'SibSp', 'Parch', 'Age', 'Fare', 'Sex', 'Embarked', 'Title'])



data_total.head()
# split and scale ========================================



data_train = data_total[:data_train_length]



X = data_train.drop(columns=['Survived'], axis=1)

y = data_train['Survived']



# first split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



# then scale both train and test data with train scaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# hyperparam optimization ========================================



"""#first round of grid search



svc_params = [{'C': [1,2,3,4,5], 'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': [.1, .2, .3, .4, .5, .6, .7, .8, .9]},

              {'C': [1,2,3,4,5], 'kernel': ['linear']}]

#output: {'C': 1, 'gamma': 0.2, 'kernel': 'rbf'}, 0.8264360018091361



knn_params = [{'n_neighbors': [3,4,5,6,7,8], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

               'leaf_size': [25, 30, 35, 40, 50], 'p': [1, 2]}]

#output: {'algorithm': 'brute', 'leaf_size': 25, 'n_neighbors': 8, 'p': 2, 'weights': 'uniform'}, 0.8294663048394391



rfc_params = [{'n_estimators': [20, 50, 100, 150, 200], 'criterion': ['gini', 'entropy']}]

#output: {'criterion': 'entropy', 'n_estimators': 100}, 0.7979873360470375

"""



svc_params = [{'C': [0.2, 0.4, 0.5, 1, 2, 3, 4, 5], 'kernel': ['rbf'], 'gamma': [.10, .15, .20, .25, .30]}]

knn_params = [{'n_neighbors': [7, 8, 9, 10], 'weights': ['uniform'], 'algorithm': ['brute'], 'p': [2]}]

rfc_params = [{'n_estimators': [80, 90, 100, 110, 120, 130, 140], 'criterion': ['entropy']}]



classifier_names = ['SVC', 'KNN', 'RFC']

params = [svc_params, knn_params, rfc_params]

classifier_objs = [SVC(), KNeighborsClassifier(), RandomForestClassifier()]

best_params = {}



for i in range(len(classifier_names)):

    classifier_objs[i].fit(X_train, y_train)

    grid_search = GridSearchCV(estimator=classifier_objs[i], param_grid=params[i], scoring='accuracy', cv=10).fit(X_train, y_train)

    best_params[classifier_names[i]] = grid_search.best_params_
# model fit and vote for predictions ========================================



# initialize classifier with best params from grid search

classifier_tuned = [SVC(C=best_params['SVC']['C'], gamma=best_params['SVC']['gamma'], kernel=best_params['SVC']['kernel']),

        KNeighborsClassifier(algorithm=best_params['KNN']['algorithm'], n_neighbors=best_params['KNN']['n_neighbors'],

                             p=best_params['KNN']['p'], weights=best_params['KNN']['weights']),

        LogisticRegression(),

        RandomForestClassifier(criterion=best_params['RFC']['criterion'], n_estimators=best_params['RFC']['n_estimators']),

        GaussianNB()]



# fit every classifier to the data and collect predictions

predictions_train = []



for clf in classifier_tuned:

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    predictions_train.append(y_pred)

    print(clf)

    print(confusion_matrix(y_test, y_pred))

    print('accuracy score: ' + str(accuracy_score(y_test, y_pred)))

    print('cross val score: ' + str(cross_val_score(clf, X_train, y_train).mean()))

    print('–––––––––')



# compute the voting result of each prediction

predictions_train = pd.DataFrame(predictions_train).transpose()

predictions_train['y_pred_voted'] = predictions_train.mean(axis=1).round()

predictions_train_voted = predictions_train['y_pred_voted'].values



print('Result after voting:')

print(confusion_matrix(y_test, predictions_train_voted))

print(accuracy_score(y_test, predictions_train_voted))
# predict test dataset



passengerIds = data_test['PassengerId']

data_test = data_total[data_train_length:].drop(columns=['Survived']).reset_index(drop=True)



scaler = StandardScaler().fit(X)

X = scaler.transform(X)

data_test = scaler.transform(data_test)



predictions_test = []

for clf in classifier_tuned:

    clf.fit(X, y)

    y_pred = clf.predict(data_test)

    predictions_test.append(y_pred)

    print(clf)

    print('score: ' + str(cross_val_score(clf, X, y).mean()))

    print('–––––––––')



predictions_test = pd.DataFrame(predictions_test).transpose()

predictions_test['Survived'] = predictions_test.mean(axis=1).round().astype(int)

predictions_test_voted = predictions_test['Survived']



submission = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions_test_voted})

submission.to_csv('gridsearch_voting_submission.csv', index=False)


