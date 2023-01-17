# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')



sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train)

    
data_train.sample(5)
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1,0,8,15,31,2000)

    group_names = ['Unknown', 'First', 'Second', 'Third', 'Forth']

    cat = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = cat

    return df

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked','Cabin'], axis=1)



def simplyfy(df):

    df = simplify_ages(df)

    df = simplify_fares(df)

    df = drop_features(df)

    return df



data_train_trans = simplyfy(data_train)

data_test_trans = simplyfy(data_test)

data_test_trans.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train_trans)
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Age', 'Sex','Fare']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test





data_train_trans, data_test_trans = encode_features(data_train_trans, data_test_trans)
from sklearn.model_selection import train_test_split



x_all = data_train_trans.drop(['Survived','PassengerId'], axis=1)

y_all = data_train_trans['Survived']

num_test = 0.20

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=25)

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



clf = svm.SVC()

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(accuracy_score(y_test, predictions))
from sklearn.cross_validation import KFold



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        x_train, x_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)
ids = data_test_trans['PassengerId']

predictions = clf.predict(data_test_trans.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

# output.to_csv('titanic-predictions.csv', index = False)

output.head()

output.to_csv('submission.csv', index=False)