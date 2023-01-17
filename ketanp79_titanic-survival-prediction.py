# Data manipulation

import numpy as np

import pandas as pd



# Data visualization

import seaborn as sb

import matplotlib.pyplot as plt



# Regex

import re as re



# Model Selection and Evaluation

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_score



# Performance

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



# Machine Learning Algorithms

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression





# Preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



# Base classes

from sklearn.base import BaseEstimator, TransformerMixin

titanic_trainSet = pd.read_csv('../input/train.csv')

titanic_testSet = pd.read_csv('../input/test.csv')

full_data = [titanic_trainSet, titanic_testSet]
titanic_trainSet.head()
titanic_testSet.head()
titanic_trainSet.info()

print('-'*40)

titanic_testSet.info()
titanic_trainSet.describe()
print(titanic_trainSet["Survived"].value_counts(sort=False))

print('-'*50)

plt.figure(figsize=(10, 6))

sb.set(style="whitegrid")

sb.countplot( x= 'Survived', hue='Sex', data=titanic_trainSet)

plt.title('Survival Distribution')

plt.show()
print("Passengers in each class\n")

titanic_trainSet["Pclass"].value_counts(sort=False)
print(titanic_trainSet[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

plt.figure(figsize=(10, 6))

sb.countplot( x= 'Pclass', hue='Survived',data=titanic_trainSet)

plt.title('Survival rate of each class')

plt.show()
print('Gender distribution\n')

print(titanic_trainSet["Sex"].value_counts(sort=False))
print(titanic_trainSet[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

plt.figure(figsize=(10, 6))

sb.countplot( x= 'Sex', hue='Survived',data=titanic_trainSet)

plt.title('Survival rate of each Gender')

plt.show()
print('Embarked distribution\n')

print(titanic_trainSet["Embarked"].value_counts(sort=False))
print(titanic_trainSet[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

plt.figure(figsize=(10, 6))

sb.countplot( x= 'Embarked', hue='Survived', data=titanic_trainSet)

plt.title('Survival rate based on boarding point')

plt.show()
for dataset in full_data:

    dataset['RelativesOnboard'] = dataset['SibSp'] + dataset['Parch']
print('Relatives distribution\n')

print(titanic_trainSet["RelativesOnboard"].value_counts(sort=False))
print(titanic_trainSet[['RelativesOnboard', 'Survived']].groupby(['RelativesOnboard'], as_index=False).mean())

plt.figure(figsize=(10, 6))

sb.countplot( x= 'RelativesOnboard', hue='Survived', data=titanic_trainSet)

plt.title('Survival rate based on no. of relatives')

plt.show()
for dataset in full_data:

    dataset['AgeGroup'] = dataset['Age'] // 15 * 15
titanic_trainSet.isna().sum()
titanic_copy = titanic_trainSet.copy()
median = titanic_copy['AgeGroup'].median()

titanic_copy['AgeGroup'] = titanic_copy['AgeGroup'].fillna(median)
print('Age distribution\n')

print(titanic_copy["AgeGroup"].value_counts(sort=False))
print(titanic_copy[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean())

plt.figure(figsize=(10, 6))

sb.countplot( x= 'AgeGroup', hue='Survived', data=titanic_copy)

plt.title('Survival rate based on Age')

plt.show()
titanic_trainSet['Name'].head()
def parseTitle(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

  

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full_data:

  dataset['Title'] = dataset['Name'].apply(parseTitle)

print('Title distribution\n')

print(titanic_trainSet["Title"].value_counts(sort=False))
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print('Title distribution\n')

print(titanic_trainSet["Title"].value_counts(sort=True))
print(titanic_trainSet[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

plt.figure(figsize=(10, 6))

sb.countplot( x= 'Title', hue='Survived', data=titanic_trainSet)

plt.title('Survival rate based on Title')

plt.show()
features_testSet = ['Pclass', 'Sex', 'Embarked', 'Fare', 'Title', 'AgeGroup', 'RelativesOnboard']

features_trainSet = features_testSet + ['Survived']



train_set = titanic_trainSet[[*features_trainSet]]

test_set = titanic_testSet[[*features_testSet]]



train_set.head()

  
test_set.head()
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median"))])

cat_pipeline = Pipeline([("imputer", CategoricalImputer()), ("cat_encoder", OneHotEncoder(sparse=False))])
num_attribs = ['Fare']

cat_attribs = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'RelativesOnboard']



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", cat_pipeline, cat_attribs),

    ])

X_train = full_pipeline.fit_transform(train_set)

y_train = train_set["Survived"]
classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True, gamma="auto"),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	  AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LogisticRegression()]
classifiers_Kscores = []

classifiers_accuracy = []

for clf in classifiers:

  clf_scores = cross_val_score(clf, X_train, y_train, cv=10)

  classifiers_Kscores.append(clf_scores)

  model_name = type(clf).__name__

  classifiers_accuracy.append(model_name+': '+str(format(clf_scores.mean()*100,'.2f')))
plt.figure(figsize=(10, 6))

plt.boxplot(classifiers_Kscores, labels=("KNN","SVC","Trees","Forest","Ada","Gradient","NB","Logistic"))

plt.ylabel("Accuracy", fontsize=14)

plt.show()



print("\n\nClassifiers Accuracy:")

classifiers_accuracy

splits = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)



for train_index, test_index in splits.split(X_train, y_train):

  X_train1, X_test1 = X_train[train_index], X_train[test_index]

  y_train1, y_test1 = y_train[train_index], y_train[test_index]


f1_scores = []

for clf in classifiers:

  clf.fit(X_train1, y_train1)

  pred = clf.predict(X_test1)

  model_name = type(clf).__name__

  f1_scores.append((model_name+': '+str(format(f1_score(y_test1, pred)*100,'.2f'))))
print("F1 Scores:")

f1_scores
X_test = full_pipeline.fit_transform(test_set)
ada_clf = AdaBoostClassifier()

ada_clf.fit(X_train,y_train)

y_pred = ada_clf.predict(X_test)
passengerID =np.array(titanic_testSet["PassengerId"]).astype(int)

titanicSurvival_predictions = pd.DataFrame(y_pred, passengerID, columns = ["Survived"])



titanicSurvival_predictions.to_csv("Titanic_Survival_Predictions_ada.csv", index_label = ["PassengerId"])