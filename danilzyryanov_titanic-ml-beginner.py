# Import libs



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Create dataframe

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



train.sample(3)
import sklearn

print (sklearn.__version__)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,

              palette={"male": "blue", "female": "red"},

              markers=["|", "o"], linestyles=["-", "--"]);
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 40, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

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

train.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=train);
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train);
sns.barplot(x="Fare", y="Survived", hue="Sex", data=train);
from sklearn import preprocessing

def encode_features(train, test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    combined = pd.concat([train[features], test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(combined[feature])

        train[feature] = le.transform(train[feature])

        test[feature] = le.transform(test[feature])

    return train, test

    

train, test = encode_features(train, test)

train.head()
from sklearn.model_selection import train_test_split



Xall = train.drop(['Survived', 'PassengerId'], axis=1)

yall = train['Survived']



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
from sklearn.model_selection import KFold



def runkfold(clf):

    kf = KFold(n_splits=10)

    outcomes = []

    fold = 0

    for trainindex, testindex in kf.split(Xall, yall):

        fold += 1

        Xtrain, Xtest = Xall.values[trainindex], Xall.values[testindex]

        ytrain, ytest = yall.values[trainindex], yall.values[testindex]

        clf.fit(Xtrain, ytrain)

        predictions = clf.predict(Xtest)

        accuracy = accuracy_score(ytest, predictions)

        outcomes.append(accuracy)

        print('Fold {0} accuracy: {1}'.format(fold, accuracy))

    mean_outcome = np.mean(outcomes)

    print("Mean Acurracy: {0}".format(mean_outcome))



runkfold(clf)
ids = test['PassengerId']

predictions = clf.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index = False)

output.head(50)