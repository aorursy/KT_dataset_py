import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

PassengerId = titanic_test['PassengerId']

titanic_train.head()
titanic_train.info()
titanic_train.describe()
titanic_train.hist(bins=50, figsize=(20, 15))

plt.show
titanic_train.info()
mean = titanic_train['Age'].mean()

titanic_train['Age'].fillna(mean, inplace=True)



mode = titanic_train['Embarked'].mode()

titanic_train['Embarked'].fillna(mode[0], inplace=True)
titanic_test.info()
age_mean = titanic_test['Age'].mean()

titanic_test['Age'].fillna(age_mean, inplace=True)



fare_mean = titanic_test['Fare'].mean()

titanic_test['Fare'].fillna(fare_mean, inplace=True)



mode = titanic_test['Embarked'].mode()

titanic_test['Embarked'].fillna(mode[0], inplace=True)
corr_matrix = titanic_train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
#  Custom Transformers to Add Extra Features¶

from sklearn.base import BaseEstimator, TransformerMixin



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, X=None, tittle = True, deck = True, family_size = True, age_class = True, fare_per_person = True):

        self.X = None

        self.tittle = tittle

        self.deck = deck

        self.family_size = family_size

        self.age_class = age_class

        self.fare_per_person = fare_per_person

    

    def substrings_in_string(self, big_string, substrings):

        for substring in substrings:

            if big_string.find(substring) != -1:

                return substring

        return substrings[-1]



    def replace_titles(self, x):

        title = x['Title']

        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

            return 'Mr'

        elif title in ['Countess', 'Mme']:

            return 'Mrs'

        elif title in ['Mlle', 'Ms']:

            return 'Miss'

        elif title =='Dr':

            if x['Sex']=='Male':

                return 'Mr'

            else:

                return 'Mrs'

        else:

            return title



    def add_tittle(self):

        title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                      'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                      'Don', 'Jonkheer']

        

        self.X['Title'] = self.X['Name'].map(lambda x: self.substrings_in_string(x, title_list))

        self.X['Title'] = self.X.apply(self.replace_titles, axis=1)

    

    def add_deck(self):

        cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

        self.X['Cabin'] = self.X['Cabin'].astype(str)

        self.X['Deck'] = self.X['Cabin'].map(lambda x: self.substrings_in_string(x, cabin_list))

    

    def add_family_size(self):

        self.X['Family_Size'] = self.X['SibSp'] + self.X['Parch']

    

    def add_age_class(self):

        self.X['Age*Class'] = self.X['Age'] * self.X['Pclass']

    

    def add_age_calss(self):

        self.X['Fare_Per_Person'] = self.X['Fare']/(self.X['Family_Size']+1)

        

    def fit(self, X, y=None):

        return self # nothing else to do

    

    def transform(self, X):

        self.X = X

        if self.tittle:

            self.add_tittle()

        if self.deck:

            self.add_deck()

        if self.family_size:

            self.add_family_size()

        if self.age_class:

            self.add_age_class()

        if self.fare_per_person:

            self.add_age_calss()

        return self.X
add_atr = CombinedAttributesAdder()

titanic_train = add_atr.transform(titanic_train)

titanic_test = add_atr.transform(titanic_test)
titanic_train.head()
titanic_train = titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

titanic_test = titanic_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



# titanic_train = pd.get_dummies(titanic_train)

# titanic_test = pd.get_dummies(titanic_test)



titanic_train.head()
from sklearn.preprocessing import OneHotEncoder



cat_features = ['Sex', 'Embarked', 'Title', 'Deck']



train_cat_features = titanic_train[cat_features]

test_cat_features = titanic_test[cat_features]



cat_encoder = OneHotEncoder()



cat_encoder.fit(train_cat_features)

train_hot_enc = cat_encoder.transform(train_cat_features).toarray()

test_hot_enc = cat_encoder.transform(test_cat_features).toarray()



train_num = titanic_train.drop(cat_features, axis=1)

test_num = titanic_test.drop(cat_features, axis=1)
y = train_num['Survived']

train_num = train_num.drop(['Survived'], axis=1)
train = np.concatenate((train_num, train_hot_enc), axis=1)

test = np.concatenate((test_num, test_hot_enc), axis=1)
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

X = std.fit_transform(train)

X_test = std.fit_transform(test)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



svc = SVC(random_state=0)

svc.fit(X_train, y_train)

train_acc = svc.score(X_train, y_train)

test_acc = svc.score(X_val, y_val)

print("========== SVC ==========")

print("Train set accuracy: {}".format(train_acc))

print("Validation set accuracy: {}".format(test_acc))



SGD = SGDClassifier(random_state=0)

SGD.fit(X_train, y_train)

train_acc = SGD.score(X_train, y_train)

test_acc = SGD.score(X_val, y_val)

print("========== SGD ==========")

print("Train set accuracy: {}".format(train_acc))

print("Validation set accuracy: {}".format(test_acc))



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)

train_acc = tree.score(X_train, y_train)

test_acc = tree.score(X_val, y_val)

print("========== Tree ==========")

print("Train set accuracy: {}".format(train_acc))

print("Validation set accuracy: {}".format(test_acc))



forest_clf = RandomForestClassifier(random_state=0)

forest_clf.fit(X_train, y_train)

print("========== Forest ==========")

train_acc = forest_clf.score(X_train, y_train)

test_acc = forest_clf.score(X_val, y_val)

print("Train set accuracy: {}".format(train_acc))

print("Validation set accuracy: {}".format(test_acc))





boost = AdaBoostClassifier(tree,random_state=0)

boost.fit(X_train, y_train)

train_acc = boost.score(X_train, y_train)

test_acc = boost.score(X_val, y_val)

print("========== Boost ==========")

print("Train set accuracy: {}".format(train_acc))

print("Validation set accuracy: {}".format(test_acc))
from sklearn.metrics import classification_report

y_pred = forest_clf.predict(X_val)

print(classification_report(y_val, y_pred, target_names=['Not-survived', 'survived']))
test_submitted = pd.Series(forest_clf.predict(X_test), name="Survived")



results = pd.concat([PassengerId,test_submitted],axis=1)



results.to_csv("Gender_Submission.csv",index=False)