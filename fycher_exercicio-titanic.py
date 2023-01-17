import os
print(os.listdir('../input/titanic'))

PATH = '../input/titanic/'
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



plt.style.use('seaborn-darkgrid')
train = pd.read_csv(PATH + 'train.csv')
train.head()
train.info()
train['PassengerId'].value_counts().shape, train['Ticket'].value_counts().shape
train.drop(columns=['PassengerId', 'Ticket'], axis=0, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns='Survived', axis=1), train['Survived'],

                                                    test_size=0.2, stratify=train['Survived'], random_state=42)
X_train.isna().sum()
y_train.value_counts()
def fill_cabin(dataframe):

    dataframe['Cabin'] = dataframe['Cabin'].fillna('Z').apply(lambda x: x[0])
fill_cabin(X_train)

fill_cabin(X_test)
X_train['Cabin'].value_counts()
def replace_cabin(dataframe):

    l = list(dataframe['Cabin'].value_counts()[dataframe['Cabin'].value_counts() < 10].index)

    dataframe['Cabin'].replace(l, 'X', inplace=True)
replace_cabin(X_train)

replace_cabin(X_test)
X_train['Cabin'].value_counts()
embarked_mode = train['Embarked'].mode()[0]

def fill_embarked(dataframe):

    dataframe['Embarked'].fillna(embarked_mode, inplace=True)
fill_embarked(X_train)

fill_embarked(X_test)
X_train['Embarked'].value_counts()
sex_dict = {'male': 0, 'female': 1}

def map_sex(dataframe):

    dataframe['Sex'] = dataframe['Sex'].map(sex_dict)
map_sex(X_train)

map_sex(X_test)
def get_title(dataframe):

    dataframe['Name'] = dataframe['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
get_title(X_train)

get_title(X_test)
X_train['Name'].value_counts()
titles = {

    'Mr': 'Mr',

    'Miss': 'Miss',

    'Mrs': 'Mrs',

    'Master': 'Master',

    'Dr': 'Dr',

    'Rev': 'Rev',

    'Col': 'Officer',

    'Mlle': 'Miss',

    'Major': 'Officer',

    'Ms': 'Mrs',

    'Capt': 'Officer',

    'Lady': 'Royal',

    'the Countess': 'Royal',

    'Jonkheer': 'Royal',

    'Mme': 'Mrs',

    'Don': 'Royal',

    'Dona': 'Royal',

    'Sir': 'Royal'

}
def map_name(dataframe):

    dataframe['Name'] = dataframe['Name'].map(titles)
map_name(X_train)

map_name(X_test)
name_age = X_train[['Name', 'Age']].groupby(['Name'])



def set_missing_age(dataframe):

        dataframe['Missing'] = dataframe['Age'].isna()



def age_by_name(dataframe):

    if dataframe['Missing']:

        return name_age.get_group(dataframe['Name']).mean()[0]

    else:

        return dataframe['Age']



mean_age = train['Age'].mean()

def fill_age(dataframe):

#     dataframe['Age'].fillna(mean_age, inplace=True)

    set_missing_age(dataframe)

    dataframe['Age'] = dataframe.apply(age_by_name, axis=1)

    dataframe.drop(columns='Missing', axis=1, inplace=True)
name_age.mean()
fill_age(X_train)

fill_age(X_test)
X_train.isna().sum()
X_train.info()
from sklearn.preprocessing import LabelEncoder

cabin_encoder = LabelEncoder()

embarked_encoder = LabelEncoder()

name_encoder = LabelEncoder()
cabin_encoder.fit(X_train['Cabin'])

embarked_encoder.fit(X_train['Embarked'])

name_encoder.fit(X_train['Name'])



def encode_nominal(dataframe):

    dataframe['Cabin'] = cabin_encoder.transform(dataframe['Cabin'])

    dataframe['Embarked'] = embarked_encoder.transform(dataframe['Embarked'])

    dataframe['Name'] = name_encoder.transform(dataframe['Name'])
cabin_encoder.classes_
encode_nominal(X_train)

encode_nominal(X_test)
X_train['Cabin'].head()
cabin_encoder.inverse_transform(X_train['Cabin'].head())
X_train.info()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree
from sklearn.model_selection import StratifiedKFold
k = 5

kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)



scores = np.zeros(k)



for idx, (idx_train, idx_val) in enumerate(kfold.split(X_train, y_train)):

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    clf.fit(X_train.iloc[idx_train], y_train.iloc[idx_train])

    scores[idx] = clf.score(X_train.iloc[idx_val], y_train.iloc[idx_val])



print("Acurácia por split: {}".format(scores))

print("Acurácia média: {}".format(scores.mean()))
clf = DecisionTreeClassifier(random_state=42, max_depth=5)

clf.fit(X_train, y_train)
plt.figure(dpi=400)

plot_tree(clf, filled=True, feature_names=X_train.columns);
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf.feature_importances_})

feature_importance.sort_values('Importance', inplace=True)
feature_importance.plot.barh(x='Feature', y='Importance', figsize=(10,8));
clf.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV
n_features = X_train.shape[1]



params = {

    'max_depth': range(1, n_features + 1),

    'criterion': ['gini', 'entropy'],

    'min_samples_split': range(2, 10),

    'min_samples_leaf': range(1, 10),

    'max_features': range(1, n_features + 1),

    'class_weight': ['balanced', None]

}



grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 

                    param_grid=params, n_jobs=-1, cv=3, verbose=2, iid=False)
grid.fit(X_train, y_train)

grid.best_score_
grid.best_params_
fill_cabin(train)

replace_cabin(train)

fill_embarked(train)

map_sex(train)

get_title(train)

map_name(train)

fill_age(train)

encode_nominal(train)
X, y = train.drop(columns=['Survived'], axis=1), train['Survived']
final_clf = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',

                                   max_depth=9, max_features=4, min_samples_leaf=1,

                                   min_samples_split=7)
final_clf.fit(X, y)
test = pd.read_csv(PATH + 'test.csv')
test.info()
test.head()
ids = test['PassengerId']

test.drop(columns=['PassengerId', 'Ticket'], axis=1, inplace=True)
fill_cabin(test)

replace_cabin(test)

fill_embarked(test)

map_sex(test)

get_title(test)

map_name(test)

fill_age(test)

encode_nominal(test)
test.head()
test.isna().sum()
test['Fare'].fillna(0, inplace=True)
survived = final_clf.predict(test.values)
answer = pd.DataFrame({'PassengerId': ids, 'Survived': survived})
answer.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()