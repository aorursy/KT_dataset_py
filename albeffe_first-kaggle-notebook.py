# Import ------------------------------------------------------------------------------------------

import csv

import pandas as pd

import sys

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, ParameterGrid

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC
# Functions ---------------------------------------------------------------------------------------

def custom_csv_print(in_id, in_labels):

    list_to_print = []

    for index in range(0, len(in_labels)):

        row_to_print = [in_id[index], in_labels[index]]



        list_to_print.append(row_to_print)



    with open('../output/result.csv', 'w', newline='', encoding='utf-8') as f:

        writer = csv.writer(f)

        writer.writerow(['PassengerId', 'Survived'])

        for index in range(0, len(list_to_print)):

            writer.writerow(list_to_print[index])

    return
# Main --------------------------------------------------------------------------------------------

df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")



# Print training dataset info

print(f'Number of passengers: {len(df_train)}')

print(f'Columns: {df_train.columns}')

print()



# Check missing labels

if df_train['Survived'].isna().sum() > 0:

    print('Missing Labels')

    sys.exit(1)



# Check NaN values

print(f'Pclass Nan values: {df_train["Pclass"].isna().sum()}')

print(f'Sex Nan values: {df_train["Sex"].isna().sum()}')

print(f'Age Nan values: {df_train["Age"].isna().sum()}')

print(f'SibSp Nan values: {df_train["SibSp"].isna().sum()}')

print(f'Parch Nan values: {df_train["Parch"].isna().sum()}')

print(f'Ticket Nan values: {df_train["Ticket"].isna().sum()}')

print(f'Fare Nan values: {df_train["Fare"].isna().sum()}')

print(f'Cabin Nan values: {df_train["Cabin"].isna().sum()}')

print(f'Embarked Nan values: {df_train["Embarked"].isna().sum()}')

print()
# Merging Train and Test Datasets before Pre-processing

df_train_labels = df_train['Survived']

df_train = df_train.drop(columns=['Survived'])

df_tot = df_train.append(df_test)

df_tot_PassengerId = df_tot['PassengerId']



# Dropping non-relevant columns

df_tot = df_tot.drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin', 'Fare'])

print(f'Columns: {df_tot.columns}')

print()



# Sex column conversion

df_tot.loc[df_tot["Sex"] == 'male', "Sex"] = 1

df_tot.loc[df_tot["Sex"] == 'female', "Sex"] = 0



# Replacing Age NaN with mean value age

df_tot['Age'].fillna(df_tot['Age'].mean(), inplace=True)



# Dividing Train and Test Datasets

df_train_unlabeled = df_tot[:len(df_train)]

df_test_processed = df_tot[len(df_train):]



X_train, X_test, y_train, y_test = train_test_split(df_train_unlabeled, df_train_labels, test_size=0.25, random_state=1)

max_accuracy = 0
# RandomForestClassifier --------------------------------------------------------------------------

randomforest_flag = False



if randomforest_flag:

    hyp_parameters = {"max_depth": [None, 5, 10],

                      "max_features": ['auto', 'sqrt'],

                      "min_samples_split": [2, 3, 10],

                      "min_samples_leaf": [1, 3, 10],

                      "bootstrap": [False],

                      "n_estimators": [100, 1000],

                      "criterion": ["gini"]}



    config_cnt = 0

    tot_config = 3 * 2 * 3 * 3 * 2



    for config in ParameterGrid(hyp_parameters):

        config_cnt += 1

        print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')



        clf = RandomForestClassifier(**config)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        clf_accuracy = accuracy_score(y_test, y_pred)



        if clf_accuracy > max_accuracy:

            max_accuracy = clf_accuracy

            print(f"-----> Accuracy: {clf_accuracy}")

            print()
# SVC ---------------------------------------------------------------------------------------------

svc_flag = False



if svc_flag:

    hyp_parameters = {'kernel': ['rbf', 'linear'],

                      'gamma': [0.001, 0.01, 0.1, 1],

                      'C': [1, 10, 50, 100, 200, 300, 1000]}



    config_cnt = 0

    tot_config = 2 * 4 * 7



    for config in ParameterGrid(hyp_parameters):

        config_cnt += 1

        print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')



        clf = SVC(**config)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        clf_accuracy = accuracy_score(y_test, y_pred)



        if clf_accuracy > max_accuracy:

            max_accuracy = clf_accuracy

            print(f"-----> Accuracy: {clf_accuracy}")

            print()
# SGD ---------------------------------------------------------------------------------------------

sgd_flag = False



if sgd_flag:

    hyp_parameters = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],

                      'alpha': [0.00001, 0.0001, 0.001],

                      'penalty': ['l2', 'l1', 'elasticnet']}



    config_cnt = 0

    tot_config = 5 * 3 * 3



    for config in ParameterGrid(hyp_parameters):

        config_cnt += 1

        print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')



        clf = SGDClassifier(**config)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        clf_accuracy = accuracy_score(y_test, y_pred)



        if clf_accuracy > max_accuracy:

            max_accuracy = clf_accuracy

            print(f"-----> Accuracy: {clf_accuracy}")

            print()
# Output ------------------------------------------------------------------------------------------

final_clf = SVC(C=200, gamma=0.001, kernel='rbf')

final_clf.fit(df_train_unlabeled, df_train_labels)

final_y_pred = final_clf.predict(df_test_processed)



custom_csv_print(df_tot_PassengerId[len(df_train):].values, final_y_pred)