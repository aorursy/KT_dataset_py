# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Fare'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

test_df.head()
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Family'] = dataset['SibSp'] + dataset['Parch']
    dataset['Pclass2'] = dataset['Pclass'].map({1:'P1', 2:'P2', 3:'P3'})
    

guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.drop('Pclass',axis=1)
test_df.drop('Pclass',axis=1)
train_df.head()
one_hot_encoded_training_predictors = pd.get_dummies(train_df)
one_hot_encoded_test_predictors = pd.get_dummies(test_df)

one_hot_encoded_training_predictors.fillna(0)
one_hot_encoded_test_predictors.fillna(0)


one_hot_encoded_training_predictors.head()

X_train = one_hot_encoded_training_predictors.drop("Survived", axis=1)
Y_train = one_hot_encoded_training_predictors["Survived"]
X_test  = one_hot_encoded_test_predictors.drop('PassengerId', axis=1).copy()

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = cross_val_score(logreg, X_train, Y_train, cv=5) 
np.average(acc_random_forest)
#X_test.head()
X_test.head()
Y_pred = random_forest.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)
