!pip install pandas scikit-learn matplotlib
import pandas

titanic_train = pandas.read_csv('../input/train.csv')

titanic_train
titanic_train.describe()
titanic_train.info()
%matplotlib inline

import matplotlib.pyplot as plt

titanic_train.hist(bins=50, figsize=(20,15))

plt.show()
# Create two datasets for comparison - survivors and rest

titanic_survivors = titanic_train[titanic_train['Survived'] == 1].copy()

titanic_casualties = titanic_train[titanic_train['Survived'] == 0].copy()
titanic_survivors
titanic_train[titanic_train['Cabin'] == 'C123']
titanic_casualties
titanic_train[titanic_train['Name'].str.contains('Johnson')]
# Create a new column with age strata

import numpy as np

from collections import OrderedDict

import itertools



age_categories = OrderedDict({'child': [0, 10],

                  'teen': [10, 20],

                  'adult': [20, 40],

                  'middle_age': [40, 60],

                  'old': [60, np.inf]})



titanic_survivors["age_category"] = pandas.cut(titanic_survivors["Age"],

                                    bins=list(itertools.chain.from_iterable(age_categories.values())),

                                    labels=age_categories.keys(), duplicates='drop')

titanic_casualties["age_category"] = pandas.cut(titanic_casualties["Age"],

                                    bins=list(itertools.chain.from_iterable(age_categories.values())),

                                    labels=age_categories.keys(), duplicates='drop')
# Compare survivors vs casualties based on Sex

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

titanic_casualties.Sex.value_counts().plot(kind='pie', ax=axes[0], label='Casualties', colors=['blue', 'orange'])

titanic_survivors.Sex.value_counts().plot(kind='pie', ax=axes[1], label='Survivors', colors=['orange', 'blue'])

fig.show()
# Compare survivors vs casualties based on age_class

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

titanic_casualties.age_category.value_counts().plot(kind='pie', ax=axes[0], label='Casualties')

titanic_survivors.age_category.value_counts().plot(kind='pie', ax=axes[1], label='Survivors')

fig.show()
# Compare survivors vs casualties based on age_class and sex

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

titanic_casualties[titanic_casualties['Sex'] == 'female'].age_category.value_counts().plot(kind='pie', ax=axes[0, 0], label='Female Casualties')

titanic_survivors[titanic_survivors['Sex'] == 'female'].age_category.value_counts().plot(kind='pie', ax=axes[0, 1], label='Female Survivors')

titanic_casualties[titanic_casualties['Sex'] == 'male'].age_category.value_counts().plot(kind='pie', ax=axes[1, 0], label='Male Casualties')

titanic_survivors[titanic_survivors['Sex'] == 'male'].age_category.value_counts().plot(kind='pie', ax=axes[1, 1], label='Male Survivors')



fig.show()
# Compare survivors vs casualties based on number of parents/children

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,15))

titanic_casualties.Parch.value_counts().plot(kind='pie', ax=axes[0], label='Casualties')

titanic_survivors.Parch.value_counts().plot(kind='pie', ax=axes[1], label='Survivors')

fig.show()
# Compare survivors vs casualties based on number of siblings/spouses

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,15))

titanic_casualties.SibSp.value_counts().plot(kind='pie', ax=axes[0], label='Casualties')

titanic_survivors.SibSp.value_counts().plot(kind='pie', ax=axes[1], label='Survivors')

fig.show()
# Compare survivors vs casualties based on passenger class

#titanic_casualties['CabinOwner'] = 



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,15))

titanic_casualties.Pclass.value_counts().plot(kind='pie', ax=axes[0], label='Casualties')

titanic_survivors.Pclass.value_counts().plot(kind='pie', ax=axes[1], label='Survivors')

fig.show()
# Compare survivors vs casualties based on passenger class

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,15))

titanic_casualties.Embarked.value_counts().plot(kind='pie', ax=axes[0], label='Casualties')

titanic_survivors.Embarked.value_counts().plot(kind='pie', ax=axes[1], label='Survivors')

fig.show()
# Compare survivors vs casualties based on having a cabin

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,15))

titanic_casualties.Embarked.value_counts().plot(kind='pie', ax=axes[0], label='Casualties')

titanic_survivors.Embarked.value_counts().plot(kind='pie', ax=axes[1], label='Survivors')

fig.show()
# Compare survivors vs casualties based on fare

titanic_casualties.Fare.plot(kind='hist', label='Casualties', color='red', alpha=0.5)

titanic_survivors.Fare.plot(kind='hist', label='Survivors', color='blue', alpha=0.5)
# Check if we can create family and cabin location based clusters of survivors and casualties



def family_survived(X):

    X
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin



prepared_titanic_train = titanic_train.copy()



def has_cabin(row):

    if type(row['Cabin']) is float and np.isnan(row['Cabin']):

        return 0.0

    else:

        return 1.0



    

def impute_embarked(row):

    if row['Embarked'] not in {'Q', 'S', 'C'}:

        return 'S'

    else:

        return row['Embarked']

    

    

def is_female(row):

    if row['Sex'] == 'female':

        return 1.0

    else:

        return 0.0

    

    

def with_family(row):

    if row['Parch'] > 0 or row['SibSp'] > 0:

        return 1.0

    else:

        return 0.0



    

    

# Transformation pipeline

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



numeric_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('scaler', StandardScaler()),

    ])



numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch']

categorical_columns = ['Pclass', 'Embarked']

drop_columns = ['PassengerId', 'Ticket', 'Name', 'Cabin', 'Sex', 'WithFamily']

pass_columns = ['CabinOwner', 'IsFemale']



full_pipeline = ColumnTransformer([

        ('drop', 'drop', drop_columns),

        ('num', numeric_pipeline, numerical_columns),

        ('cat', OneHotEncoder(), categorical_columns),

        ('pass', 'passthrough', pass_columns)

        ])



def pipeline_fit_transform(X):

    X['CabinOwner'] = X.apply(lambda row: has_cabin(row), axis=1)

    X['Embarked'] = X.apply(lambda row: impute_embarked(row), axis=1)

    X['IsFemale'] = X.apply(lambda row: is_female(row), axis=1)

    X['WithFamily'] = X.apply(lambda row: with_family(row), axis=1)

    return full_pipeline.fit_transform(X)



def pipeline_transform(X):

    X['CabinOwner'] = X.apply(lambda row: has_cabin(row), axis=1)

    X['Embarked'] = X.apply(lambda row: impute_embarked(row), axis=1)

    X['IsFemale'] = X.apply(lambda row: is_female(row), axis=1)

    X['WithFamily'] = X.apply(lambda row: with_family(row), axis=1)

    return full_pipeline.transform(X)



print(prepared_titanic_train.describe())

prepared_titanic_labels = prepared_titanic_train['Survived'].copy()

prepared_titanic_train = prepared_titanic_train.drop('Survived', axis=1)

prepared_titanic_train = pipeline_fit_transform(prepared_titanic_train)

train_set, val_set, train_labels, val_labels = train_test_split(prepared_titanic_train, prepared_titanic_labels, test_size=0.2, random_state=42)
# Get some baseline score

from sklearn.dummy import DummyClassifier



dummy_classifier = DummyClassifier()

dummy_classifier.fit(train_set, train_labels)

dummy_classifier.score(val_set, val_labels)
# Check default decision tree score

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(train_set, train_labels)

decision_tree.score(val_set, val_labels)
# Check default random forest score

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=50)

random_forest.fit(train_set, train_labels)

random_forest.score(val_set, val_labels)
transformed_columns = list(numerical_columns)

cat_encoder = full_pipeline.named_transformers_['cat']

for category in cat_encoder.categories_:

    transformed_columns.extend(category)

transformed_columns.extend(pass_columns)



feature_importances = pandas.DataFrame(random_forest.feature_importances_,

                                       index = transformed_columns,

                                       columns=['importance']).sort_values('importance', ascending=False)

feature_importances
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier(loss='log')
from sklearn.svm import SVC

svm_classifier = SVC(gamma='auto')
import xgboost as xgb



xgb_classifier = xgb.XGBClassifier(objective='binary:logistic')
# Generic cross validation model checking

from sklearn.model_selection import cross_val_score



def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



models = [('decision_tree', decision_tree), ('random_forest', random_forest), ('sgd', sgd), ('svm', svm_classifier), ('xgb', xgb_classifier)]



for model_name, model in models:

    scores = cross_val_score(model, prepared_titanic_train, prepared_titanic_labels, cv=5)

    print(f'{model_name} scores:')

    display_scores(scores)

    print()
# Random search hyperparameters for randomforest



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint



param_dist = {"n_estimators": [3, 200],

              "max_features": sp_randint(3, 11),

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}



randomized_search = RandomizedSearchCV(random_forest, param_dist, cv=5, return_train_score=True)



randomized_search.fit(prepared_titanic_train, prepared_titanic_labels)

print(randomized_search.best_estimator_)

print(randomized_search.best_score_)
# Grid search hyperparameters for randomforest



from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30, 60, 100], 'max_features': [4, 6, 8, 11]}

  ]



grid_search = GridSearchCV(random_forest, param_grid, cv=5,

                           return_train_score=True)



grid_search.fit(prepared_titanic_train, prepared_titanic_labels)

grid_search

print(grid_search.best_estimator_)

print(grid_search.best_score_)
from sklearn.model_selection import GridSearchCV

# Grid search hyperparameters for xgboost



param_grid = [

    {'n_estimators': [25, 50, 100, 200, 300], 'max_depth': [3, 6, 9, 12]}

  ]



grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5,

                           return_train_score=True)

grid_search.fit(prepared_titanic_train, prepared_titanic_labels)

best_xgboost = grid_search.best_estimator_

print(best_xgboost)

print(grid_search.best_score_)
eval_set = [(train_set,train_labels),(val_set,val_labels)]

best_xgboost.fit(train_set,train_labels,eval_metric=['error','logloss','auc'],eval_set=eval_set,verbose=True)

best_xgboost.score(val_set, val_labels)
titanic_test = pandas.read_csv('../input/test.csv')

prepared_titanic_test = pipeline_transform(titanic_test)

titanic_test
predictions = best_xgboost.predict(prepared_titanic_test)

predictions
submission = pandas.DataFrame({ 'PassengerId': titanic_test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)