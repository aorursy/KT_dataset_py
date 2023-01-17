# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_path = '../input/train.csv'

test_path = '../input/test.csv'
train_df = pd.read_csv(train_path)

train_df.info()
train_df.describe()
train_df.head()
train_df['Sex'] = train_df['Sex'].replace('male', 0).replace('female', 1)
train_df.columns
columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare']

# Come back for cabin and embarked

train_df = train_df[columns]

train_df = train_df.fillna(train_df.mean())

train_df.head()
train_df.columns
columns_X = [x for x in columns if x != 'Survived']

X = train_df[columns_X]

y = train_df['Survived']

X.head()
from sklearn.model_selection import train_test_split



class Tester:

    def __init__(self, X, y):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)

        

    def test_model(self, mod):

        mod.fit(self.X_train, self.y_train)

        score = mod.score(self.X_test, self.y_test)

        print(str(mod) + ': ' + str(score))
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier



tester = Tester(X, y)

print('SVC')

tester.test_model(SVC(gamma='scale'))

tester.test_model(SVC(gamma='scale', C=10))

tester.test_model(SVC(gamma='scale', C=100))



print('LinearSVC')

tester.test_model(LinearSVC())

tester.test_model(LinearSVC(C=10))

tester.test_model(LinearSVC(C=100))



print('DecisionTreeClassifier')

tester.test_model(DecisionTreeClassifier())

tester.test_model(DecisionTreeClassifier(max_features=5))

tester.test_model(DecisionTreeClassifier(max_features=6))



print('DecisionTreeRegressor')

tester.test_model(DecisionTreeRegressor())
print('RandomForestClassifier')

tester.test_model(RandomForestClassifier(n_estimators=100))

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

tester.test_model(RandomForestClassifier(n_estimators=100, max_leaf_nodes=10))
from sklearn.neural_network import MLPClassifier

print('MLPClassifier')

tester.test_model(MLPClassifier())
def preprocess_df(df_path):

    train_df = pd.read_csv(df_path)

    train_df['Sex'] = train_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare']

    # Come back for cabin and embarked

    train_df = train_df[columns]

    train_df = train_df.fillna(train_df.mean())

    

    columns_X = [x for x in columns if x != 'Survived']

    X = train_df[columns_X]

    y = train_df['Survived']

    return X, y
def preprocess_test(df_path):

    test_df = pd.read_csv(df_path)

    ids = test_df['PassengerId']

    test_df['Sex'] = test_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    # Come back for cabin and embarked

    test_df = test_df[columns]

    test_df = test_df.fillna(test_df.mean())

    

    return test_df, ids
def format_and_save(df, ids, out):

    merged = np.asarray([ids, df]).T

    df = pd.DataFrame(merged)

    df.columns = ['PassengerId', 'Survived']

    df.to_csv(out, index=False)
X, y = preprocess_df(train_path)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

tester.test_model(MLPClassifier())
rfc = RandomForestClassifier(n_estimators=100, max_depth=10)

rfc.fit(X, y)

validation, ids = preprocess_test(test_path)

predictions = rfc.predict(validation)
format_and_save(predictions, ids, 'naive-sample.csv')
train_df[['Fare', 'Pclass']].sort_values('Fare')
def preprocess_df_fare_test(df_path, fare=True):

    train_df = pd.read_csv(df_path)

    train_df['Sex'] = train_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',

       'Parch']

    if fare:

        columns = columns + ['Fare']

    # Come back for cabin and embarked

    train_df = train_df[columns]

    train_df = train_df.fillna(train_df.mean())

    

    columns_X = [x for x in columns if x != 'Survived']

    X = train_df[columns_X]

    y = train_df['Survived']

    return X, y
X, y = preprocess_df_fare_test(train_path, fare=True)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

X, y = preprocess_df_fare_test(train_path, fare=False)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))
def preprocess_df_pclass_test(df_path, pclass=True):

    train_df = pd.read_csv(df_path)

    train_df['Sex'] = train_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Survived', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare']

    if pclass:

        columns = columns + ['Pclass']

    # Come back for cabin and embarked

    train_df = train_df[columns]

    train_df = train_df.fillna(train_df.mean())

    

    columns_X = [x for x in columns if x != 'Survived']

    X = train_df[columns_X]

    y = train_df['Survived']

    return X, y
X, y = preprocess_df_pclass_test(train_path, pclass=True)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

X, y = preprocess_df_pclass_test(train_path, pclass=False)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))
train_df_experiment = pd.read_csv(train_path)

# Percent of nan values

train_df_experiment['Cabin'].isna().sum() / len(train_df_experiment)
# Values possible

train_df_experiment['Embarked'].unique()
# Percent of nan values

train_df_experiment['Embarked'].isna().sum() / len(train_df_experiment)
train_df_experiment[['Embarked']].head()
def oh_embarked(df):

    oh_ec = pd.get_dummies(df[['Embarked']].dropna())

    df = df.drop(['Embarked'], axis=1)

    return pd.concat([df, oh_ec], axis=1)

def preprocess_df_embarked_test(df_path, embarked=True):

    train_df = pd.read_csv(df_path)

    train_df['Sex'] = train_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Survived', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare', 'Pclass']

    if embarked:

        columns = columns + ['Embarked_C', 'Embarked_Q', 'Embarked_S']

        train_df = oh_embarked(train_df)

    

    # Come back for cabin and embarked

    train_df = train_df[columns]

    train_df = train_df.fillna(train_df.mean())

    

    columns_X = [x for x in columns if x != 'Survived']

    X = train_df[columns_X]

    y = train_df['Survived']

    return X, y
X, y = preprocess_df_embarked_test(train_path, embarked=True)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

X, y = preprocess_df_embarked_test(train_path, embarked=False)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))
print(train_df_experiment[train_df_experiment['Embarked'] == 'C']['Survived'].mean())

print(train_df_experiment[train_df_experiment['Embarked'] == 'Q']['Survived'].mean())

print(train_df_experiment[train_df_experiment['Embarked'] == 'S']['Survived'].mean())
X, y = preprocess_df_embarked_test(train_path, embarked=True)

tester = Tester(X, y)

tester.test_model(LinearSVC(C=100))

X, y = preprocess_df_embarked_test(train_path, embarked=False)

tester = Tester(X, y)

tester.test_model(LinearSVC(C=100))
def preprocess_test(df_path, ohc_embarked=False):

    test_df = pd.read_csv(df_path)

    ids = test_df['PassengerId']

    test_df['Sex'] = test_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    if ohc_embarked:

        columns = columns + ['Embarked_C', 'Embarked_Q', 'Embarked_S']

        test_df = oh_embarked(test_df)

    

    # Come back for cabin and embarked

    test_df = test_df[columns]

    test_df = test_df.fillna(test_df.mean())

    

    return test_df, ids
rfc = RandomForestClassifier(n_estimators=100, max_depth=10)

X, y = preprocess_df_embarked_test(train_path, embarked=True)

rfc.fit(X, y)

print(X.columns)

validation, ids = preprocess_test(test_path, ohc_embarked=True)

print(validation.columns)

predictions = rfc.predict(validation)

format_and_save(predictions, ids, 'ohc-embarked-sample.csv')
def dump_train_preprocess(df_path):

    train_df = pd.read_csv(df_path)

    train_df['Sex'] = train_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']

    

    # Come back for cabin and embarked

    train_df = train_df[columns]

    train_df = train_df.fillna(train_df.mean())

    

    columns_X = [x for x in columns if x != 'Survived']

    X = train_df[columns_X]

    y = train_df['Survived']

    return X, y



def dumb_test_preprocess(df_path):

    test_df = pd.read_csv(df_path)

    ids = test_df['PassengerId']

    test_df['Sex'] = test_df['Sex'].replace('male', 0).replace('female', 1)

    columns = ['Pclass', 'Sex', 'Age', 'Fare']

    

    # Come back for cabin and embarked

    test_df = test_df[columns]

    test_df = test_df.fillna(test_df.mean())

    

    return test_df, ids
X, y = dump_train_preprocess(train_path)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

tester.test_model(MLPClassifier())

X, y = dump_train_preprocess(train_path)

tester = Tester(X, y)

tester.test_model(RandomForestClassifier(n_estimators=100, max_depth=10))

tester.test_model(MLPClassifier())
rfc = RandomForestClassifier(n_estimators=100, max_depth=10)

X, y = dump_train_preprocess(train_path)

rfc.fit(X, y)

print(X.columns)

validation, ids = dumb_test_preprocess(test_path)

print(validation.columns)

predictions = rfc.predict(validation)

print(predictions)

format_and_save(predictions, ids, 'ohc-embarked-sample.csv')