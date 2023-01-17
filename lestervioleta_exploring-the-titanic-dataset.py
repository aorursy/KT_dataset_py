# Importing basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Importing the Titanic dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_file_path = '/kaggle/input/titanic/train.csv'

train_data = pd.read_csv(train_file_path)

test_file_path = '/kaggle/input/titanic/test.csv'

test_data = pd.read_csv(test_file_path)
train_data.head()
train_data.dtypes
train_data[train_data.columns[1:]].corr()['Survived'][:]
train_data.info()

print("-"*100)

test_data.info()
print(train_data['Age'].describe())

print("-"*100)

print(train_data['Fare'].describe())
import seaborn as sns
#Percentage of factor of each feature on the chances of survival
datasets = [train_data, test_data]
train_data.dropna(subset = ["Embarked"], inplace=True)
for dataset in datasets:

    dataset['Family Members'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in datasets:

    dataset['Boarded Alone'] = 0

    dataset.loc[dataset['Family Members'] == 1, 'Boarded Alone'] = 1
for dataset in datasets:

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

    dataset['Categorical Fare'] = pd.qcut(dataset['Fare'], 4)

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    dataset['Categorical Age'] = pd.qcut(dataset['Age'], 4)
def edit_cabin(dataset):

    dataset["Cabin"] = dataset["Cabin"].fillna("Suite")

    dataset.loc[~dataset["Cabin"].isin(["Suite"]), "Cabin"] = "Regular"

    return dataset



train_data = edit_cabin(train_data)

test_data = edit_cabin(test_data)
train_data.info()
import re as re



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)



    if title_search:

        return title_search.group(1)

    return ""



for dataset in datasets:

    dataset['Title'] = dataset['Name'].apply(get_title)
train_data.info()
y_train = train_data['Survived']

X_train = train_data.drop(['Survived', 'Name', 'Ticket', 'PassengerId', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1)

X_valid = test_data.drop(['Name', 'Ticket', 'PassengerId', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1)
X_train.head()
categorical_cols = ["Sex", "Embarked", "Cabin", "Categorical Fare", "Categorical Age", "Title"]
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0)



rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



rf_pipeline.fit(X_train, y_train)
submission = rf_pipeline.predict(X_valid)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': submission})

output.to_csv('lester_submission.csv', index=False)

print("Your submission was successfully saved!")