#importing packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns 

%matplotlib inline



# Importing csv files

# Input data files are available in the "../input/" directory.

train = pd.read_csv("../input/train.csv") 

test = pd.read_csv("../input/test.csv")

train.sample(5)
from sklearn import preprocessing

def encode_features(df_train, df_test): #label encoding for sklearn algorithms

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



def simplify_ages(df): # Binning ages 

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df): # Storing first letter of cabins

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df): # Binning fares

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df): # Keeping title

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df): # Dropping useless features

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



train = transform_features(train)

test = transform_features(test)

train_vis = train.copy(deep=True) # data for visualization

train, test = encode_features(train, test)

train.head()
sns.barplot(x="Sex", y="Survived", data=train_vis)
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_vis)
sns.barplot(x="Age", y="Survived", hue="Sex", data=train_vis)
sns.barplot(x="Pclass", y="Survived", hue="Fare", data=train_vis)
from sklearn.model_selection import train_test_split



X_all = train.drop(['Survived', 'PassengerId'], axis=1)

y_all = train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))
from sklearn.cross_validation import KFold



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)
ids = test['PassengerId']

predictions = clf.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

output.head()