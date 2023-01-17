import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import style

style.use('ggplot')
titanic_train = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")
titanic_train.head()
titanic_train.shape
titanic_train.info()
unique = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

unique_dict = {}

for itm in unique:

    unique_dict[itm] = titanic_train[itm].unique()
unique_dict
plt.hist(titanic_train["Fare"], bins=20)

plt.show()

plt.hist(titanic_train["Age"].dropna(), bins=10)

plt.show()
del_list = ["Name","Ticket","Cabin"]

for itm in del_list:

    del titanic_train[itm]

titanic_train.set_index("PassengerId", inplace=True)
titanic_train.head()
titanic_train.info()
titanic_train["Sex"] = titanic_train["Sex"].replace(["male","female"],[1,0])

titanic_train["Embarked"] = titanic_train["Embarked"].replace(["C","Q","S"],[1,2,3])
titanic_train.info()
titanic_train["Embarked"].value_counts()
titanic_train["Embarked"].fillna(3.0, inplace=True)

titanic_train["Embarked"] = titanic_train["Embarked"].astype(np.int64)

titanic_train.info()
age_len = titanic_train["Age"].isnull().sum()

age_mean = titanic_train["Age"].mean()

age_std = titanic_train["Age"].std()

#random number between age_mean-age_std, age_mean+age_std

rand_nos = np.random.randint(age_mean-age_std, age_mean+age_std, age_len)



fltr = np.isnan(titanic_train["Age"])

titanic_train["Age"][fltr] = rand_nos



titanic_train.info()
Pclass_dummies = pd.get_dummies(titanic_train["Pclass"])

Pclass_dummies.columns = ['Pclass_1','Pclass_2','Pclass_3']

Pclass_dummies.index = titanic_train.index

titanic_train = pd.concat([titanic_train, Pclass_dummies], axis=1)

titanic_train.drop('Pclass', axis=1, inplace=True)



Embarked_dummies = pd.get_dummies(titanic_train["Embarked"])

Embarked_dummies.columns= ['Embarked_C', 'Embarked_Q', 'Embarked_S']

Embarked_dummies.index = titanic_train.index

titanic_train = pd.concat([titanic_train, Embarked_dummies], axis=1)

titanic_train.drop('Embarked', axis=1, inplace=True)



titanic_train.head()
titanic_train["Relatives"] = titanic_train["SibSp"] + titanic_train["Parch"]

titanic_train["Relatives"][(titanic_train["Relatives"] == 0)] = 0

titanic_train["Relatives"][(titanic_train["Relatives"] != 0)] = 1

titanic_train.drop(['SibSp','Parch'], axis=1, inplace=True)
titanic_train.info()
titanic_train["Sex"] = titanic_train["Sex"].astype(np.uint8)

titanic_train["Relatives"] = titanic_train["Relatives"].astype(np.uint8)
from sklearn.model_selection import cross_val_score, KFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
features = titanic_train.columns

features = list(features)

features.remove("Survived")

features
kf = KFold(n_splits=4, shuffle=True, random_state=1)
logreg = LogisticRegression()

acc = cross_val_score(logreg, titanic_train[features], titanic_train['Survived'], scoring='accuracy', cv=kf)

logreg_acc = np.mean(acc)

print(logreg_acc)
n = range(10,300,10)

l = range(1,11)

n_l_s = []

for i in n:

    for j in l:

        temp = [i,j]

        n_l_s.append(temp)
highest_rf_acc = 0

lowesterrorcomb = list()

for comb in n_l_s:

    rf = RandomForestClassifier(n_estimators=comb[0], random_state=2, min_samples_leaf=comb[1])

    acc = cross_val_score(rf,titanic_train[features], titanic_train['Survived'], scoring='accuracy', cv=kf)

    rf_acc = np.mean(acc)

    if rf_acc > highest_rf_acc:

        highest_rf_acc = rf_acc

        lowesterrorcomb = comb



print(highest_rf_acc)

print(lowesterrorcomb)
knn = KNeighborsClassifier(n_neighbors=20)

acc = cross_val_score(knn,titanic_train[features], titanic_train['Survived'], scoring='accuracy', cv=kf)

knn_acc = np.mean(acc)

print(knn_acc)
svc = SVC(probability=True)

acc = cross_val_score(svc,titanic_train[features], titanic_train['Survived'], scoring='accuracy', cv=kf)

svc_acc = np.mean(acc)

print(svc_acc)
nb = GaussianNB()

acc = cross_val_score(nb,titanic_train[features], titanic_train['Survived'], scoring='accuracy', cv=kf)

nb_acc = np.mean(acc)

print(nb_acc)
titanic_test.shape
titanic_test.info()
unique = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

unique_dict = {}

for itm in unique:

    unique_dict[itm] = titanic_test[itm].unique()
unique_dict
plt.hist(titanic_test["Fare"].dropna(), bins=20)

plt.show()

plt.hist(titanic_test["Age"].dropna(), bins=10)

plt.show()
del_list = ["Name","Ticket","Cabin"]

for itm in del_list:

    del titanic_test[itm]

titanic_test.set_index("PassengerId", inplace=True)
titanic_test.head()
titanic_test.info()
titanic_test["Sex"] = titanic_test["Sex"].replace(["male","female"],[1,0])

titanic_test["Embarked"] = titanic_test["Embarked"].replace(["C","Q","S"],[1,2,3])
titanic_test.info()
titanic_test["Fare"].fillna(titanic_test["Fare"].mean(), inplace=True)
age_len = titanic_test["Age"].isnull().sum()

age_mean = titanic_test["Age"].mean()

age_std = titanic_test["Age"].std()

#random number between age_mean-age_std, age_mean+age_std

rand_nos = np.random.randint(age_mean-age_std, age_mean+age_std, age_len)



fltr = np.isnan(titanic_test["Age"])

titanic_test["Age"][fltr] = rand_nos
titanic_test.info()
Pclass_dummies = pd.get_dummies(titanic_test["Pclass"])

Pclass_dummies.columns = ['Pclass_1','Pclass_2','Pclass_3']

Pclass_dummies.index = titanic_test.index

titanic_test = pd.concat([titanic_test, Pclass_dummies], axis=1)

titanic_test.drop('Pclass', axis=1, inplace=True)



Embarked_dummies = pd.get_dummies(titanic_test["Embarked"])

Embarked_dummies.columns= ['Embarked_C', 'Embarked_Q', 'Embarked_S']

Embarked_dummies.index = titanic_test.index

titanic_test = pd.concat([titanic_test, Embarked_dummies], axis=1)

titanic_test.drop('Embarked', axis=1, inplace=True)
titanic_test.head()
titanic_test["Relatives"] = titanic_test["SibSp"] + titanic_test["Parch"]

titanic_test["Relatives"][(titanic_test["Relatives"] == 0)] = 0

titanic_test["Relatives"][(titanic_test["Relatives"] != 0)] = 1

titanic_test.drop(['SibSp','Parch'], axis=1, inplace=True)
titanic_test.info()
titanic_test["Sex"] = titanic_test["Sex"].astype(np.uint8)

titanic_test["Relatives"] = titanic_test["Relatives"].astype(np.uint8)
titanic_test.info()
rftest = RandomForestClassifier(n_estimators=lowesterrorcomb[0], random_state=3, min_samples_leaf=lowesterrorcomb[1])

rftest.fit(titanic_train[features], titanic_train['Survived'])

titanic_test["Survived"] = rftest.predict(titanic_test[features])
titanic_test.head()