# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_df = pd.read_csv('../input/titanic/train.csv')

titanic_df.head(3)
print('\n### learning data info ### \n')

print(titanic_df.info())
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

titanic_df['Cabin'].fillna('N', inplace=True)

titanic_df['Embarked'].fillna('N', inplace=True)

print('The Number of Null in Dataset ', titanic_df.isnull().sum().sum())
print(' The distribution of Sex values : \n', titanic_df['Sex'].value_counts())

print('\n The distribution of Cabin values : \n', titanic_df['Cabin'].value_counts())

print('\n The distribution of Embarked values : \n', titanic_df['Embarked'].value_counts())
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]

print(titanic_df['Cabin'].head(3))
titanic_df.groupby(['Sex', 'Survived'])['Survived'].count()
sns.barplot(x='Sex', y = 'Survived', data=titanic_df)
sns.barplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = titanic_df)
def get_category(age):

    

    cat = ''

    

    if age <= -1: cat = 'Unknown'

    elif age <= 5: cat = 'Baby'

    elif age <= 12: cat = 'Child'

    elif age <= 18: cat = 'Teenager'

    elif age <= 25: cat = 'Student'

    elif age <= 35: cat = 'Adult'

    elif age <= 60: cat = 'Elderly'

        

    return cat
plt.figure(figsize=(10,6))



group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Adult', 'Elderly']



titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))

sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)

titanic_df.drop('Age_cat', axis=1, inplace=True)
from sklearn import preprocessing
def encode_features(dataDF):

    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(dataDF[feature])

        dataDF[feature] = le.transform(dataDF[feature])

        

    return dataDF
titanic_df = encode_features(titanic_df)

titanic_df.head()
def fillna(df):

    df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

    df['Cabin'].fillna('N', inplace=True)

    df['Embarked'].fillna('N', inplace=True)

    df['Fare'].fillna(0, inplace=True)

    return df
def drop_features(df):

    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    return df
def format_features(df):

    df['Cabin'] = df['Cabin'].str[:1]

    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df[feature])

        df[feature] = le.transform(df[feature])

    return df
def transform_features(df):

    df = fillna(df)

    df = drop_features(df)

    df = format_features(df)

    return df
titanic_df = pd.read_csv('../input/titanic/train.csv')

y_titanic_df = titanic_df['Survived']

X_titanic_df = titanic_df.drop('Survived', axis=1)



X_titanic_df = transform_features(X_titanic_df)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_titanic_df, y_titanic_df, 

                                                 test_size=0.2, random_state=11) 
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
dt_clf = DecisionTreeClassifier(random_state=11)

rf_clf = RandomForestClassifier(random_state=11)

lf_clf = LogisticRegression()
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[2, 3, 5, 10],

             'min_samples_split':[2, 3, 5], 'min_samples_leaf':[1, 5, 8]}



grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)

grid_dclf.fit(X_train, y_train)



print('The optimal paramters of GridSearchCV :', grid_dclf.best_params_)

print('The best score of GridSearchCV : {:.4f}'.format(grid_dclf.best_score_))

best_dclf = grid_dclf.best_estimator_
dpredictions = best_dclf.predict(X_test)

accuracy = accuracy_score(y_test, dpredictions)

print('The accuracy of DecisionTreeClassifier in test dataset : {:.4f}'.format(accuracy))
test_titanic = pd.read_csv('../input/titanic/test.csv')

X_test = transform_features(test_titanic)
test_titanic = pd.read_csv('../input/titanic/test.csv')
predictions = best_dclf.predict(X_test)



output = pd.DataFrame({'PassengerId': test_titanic.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")