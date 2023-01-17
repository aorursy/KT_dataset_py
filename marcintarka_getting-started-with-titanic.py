# Basic setup
# Common imports

import numpy as np

# To plot pretty figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import os

import pandas as pd



from sklearn.model_selection import cross_val_score



plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



pd.options.mode.chained_assignment = None#fixme
CSV_PATH = "../input"





def load_train_data():

    csv_path = os.path.join(CSV_PATH, "train.csv")

    return pd.read_csv(csv_path)





def load_test_data():

    csv_path = os.path.join(CSV_PATH, "test.csv")

    return pd.read_csv(csv_path)
train = load_train_data()

test = load_test_data()

train.head(15)
train.info()
import seaborn as sns

sns.heatmap(train.copy().drop(['Name'],1).corr(),vmax=.6,cmap="RdBu_r",annot=True, square=True)
testPassenderId = test['PassengerId']

train = train.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare'], axis=1)

test = test.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare'], axis=1)
#basic data setup

from sklearn import preprocessing



lblEmbarked = preprocessing.LabelEncoder()

lblEmbarked.fit(np.unique(list(train['Embarked'].values)))

train['Embarked'] = lblEmbarked.transform(list(train['Embarked'].values))

test['Embarked'] = lblEmbarked.transform(list(test['Embarked'].values))



lblSex = preprocessing.LabelEncoder()

lblSex.fit(np.unique(list(train['Sex'].values)))

train['Sex'] = lblSex.transform(list(train['Sex'].values))

test['Sex'] = lblSex.transform(list(test['Sex'].values))



ageMean = train["Age"].mean()

train["Age"][np.isnan(train["Age"])] = ageMean

train['Age'] = train['Age'].astype(int) / 10

test["Age"][np.isnan(test["Age"])] = ageMean

test['Age'] = test['Age'].astype(int) / 10



#todo

#train['Fare'] = train['Fare'].astype(int)
#Playing with names

train['Name'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test['Name'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Name'], train['Sex'])
def merge_names(dataset):

    dataset['Name'] = dataset['Name'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Name'] = dataset['Name'].replace('Mlle', 'Miss')

    dataset['Name'] = dataset['Name'].replace('Ms', 'Miss')

    dataset['Name'] = dataset['Name'].replace('Mme', 'Mrs')





merge_names(train)

merge_names(test)

pd.crosstab(train['Name'], train['Sex'])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(train['Name'])

train['Name'] = le.transform(train['Name'])



test['Name'] = test['Name'].map(lambda s: '<unknown>' if s not in le.classes_ else s)

le.classes_ = np.append(le.classes_, '<unknown>')

test['Name'] = le.transform(test['Name'])
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test = test.copy()
# Support Vector Machines

from sklearn.svm import SVC

svc = SVC()

scores = cross_val_score(svc, X_train, Y_train, cv=5)

print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



svc.fit(X_train, Y_train)

Y_predSVM = svc.predict(X_test)

svc.score(X_train, Y_train)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=6)

scores = cross_val_score(random_forest, X_train, Y_train, cv=5)

print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



random_forest.fit(X_train, Y_train)

Y_predRF = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

scores = cross_val_score(gaussian, X_train, Y_train, cv=5)

print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



gaussian.fit(X_train, Y_train)

Y_predNB = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
#MLP

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-6,

                    hidden_layer_sizes=(5, 5, 5), random_state=1, max_iter=950)

scores = cross_val_score(mlp, X_train, Y_train, cv=5)

print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



mlp.fit(X_train, Y_train)

X_test = scaler.transform(X_test)

Y_predMLP = mlp.predict(X_test)

mlp.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": testPassenderId,

        "Survived": Y_predMLP

    })

submission.to_csv('titanic.csv', index=False)