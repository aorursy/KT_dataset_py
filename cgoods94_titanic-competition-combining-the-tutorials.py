# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# preprocessing

from fancyimpute import KNN



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train.csv')

test_raw = pd.read_csv('../input/test.csv')
train.drop(['PassengerId', 'Cabin'], axis=1, inplace = True)

test = test_raw.drop(['Cabin'], axis = 1)



test.info()
full = pd.concat([train, test])

sns.boxplot(x = "Fare", y = "Embarked", data = full)
full[full.Embarked != full.Embarked]
sns.boxplot(x = "Fare", y = "Embarked", data = full[full.Pclass == 1])

plt.axvline(x = 80, color = 'r', linewidth = 3)
train.loc[train.Embarked != train.Embarked, "Embarked"] = "C"

test.loc[test.Embarked != test.Embarked, "Embarked"] = "C"



full = pd.concat([train, test])

full.loc[(full.Fare == 80) & (full.Pclass == 1),:]

test[test.Fare != test.Fare]
imp_fare = full.loc[(full.Embarked == 'S') & (full.Pclass == 3), "Fare"].mean() 



test.loc[test.Fare != test.Fare, "Fare"] = round(imp_fare, 2)

test.loc[(test.Name == "Storey, Mr. Thomas"),:]
full = pd.concat([train, test])

full.info()
# Embarked

embark_dummies_train  = pd.get_dummies(train['Embarked'])

embark_dummies_test = pd.get_dummies(test['Embarked'])



train = train.join(embark_dummies_train)

test = test.join(embark_dummies_test)



train.drop(['Embarked'], axis = 1, inplace = True)

test.drop(['Embarked'], axis = 1, inplace = True)



# Sex



sex_dummies_train = pd.get_dummies(train['Sex'])

sex_dummies_test = pd.get_dummies(test['Sex'])



train = train.join(sex_dummies_train)

test = test.join(sex_dummies_test)



train.drop(['Sex'], axis = 1, inplace = True)

test.drop(['Sex'], axis = 1, inplace = True)



test.info()
k_train = int(np.sqrt(891))

k_test = int(np.sqrt(418))



train_features = train.drop(['Survived'], axis = 1).select_dtypes(include = [np.float, np.int])

test_features = test.select_dtypes(include = [np.float, np.int])



filled_ages_train = pd.DataFrame(KNN(k = k_train).complete(train_features)).loc[:,1]

filled_ages_test = pd.DataFrame(KNN(k = k_test).complete(test_features)).loc[:,1]



train.Age = round(filled_ages_train, 1)

test.Age = round(filled_ages_train, 1)



full = pd.concat([train, test])

full.info()
train_titles = train.Name.str.replace('(.*, )|(\\..*)', '').rename('Title')

train = train.join(train_titles)



test_titles = test.Name.str.replace('(.*, )|(\\..*)', '').rename('Title')

test = test.join(test_titles)
full = pd.concat([train, test])

full.groupby("Title").Title.count()
rare_title = ["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", 

              "Lady", "Major", "Rev", "Sir", "the Countess"]



train.loc[train.Title.isin(rare_title), "Title"] = "Rare"

test.loc[test.Title.isin(rare_title), "Title"] = "Rare"



full = pd.concat([train ,test])

full.groupby("Title").Title.count()
train.loc[train.Title.isin(["Mlle", "Ms"]), "Title"] = "Miss"

train.loc[train.Title == "Mme", "Title"] = "Mrs"



test.loc[test.Title.isin(["Mlle", "Ms"]), "Title"] = "Miss"

test.loc[test.Title == "Mme", "Title"] = "Mrs"



full = pd.concat([train ,test])

full.groupby("Title").Title.count()
title_dummies_train  = pd.get_dummies(train['Title'])

title_dummies_test = pd.get_dummies(test['Title'])



train = train.join(title_dummies_train)

test = test.join(title_dummies_test)



train.drop(['Title'], axis = 1, inplace = True)

test.drop(['Title'], axis = 1, inplace = True)



full = pd.concat([train ,test])

full.describe()
train_fsize = train.Parch + train.SibSp

train = train.join(train_fsize.rename('Fsize'))



test_fsize = test.Parch + test.SibSp

test = test.join(test_fsize.rename('Fsize'))



full = pd.concat([train ,test])

full.describe()
train_child = train.Age < 16

train = train.join(train_child.rename('Child'))



test_child = test.Age < 16

test = test.join(test_child.rename('Child'))



full = pd.concat([train ,test])

full.groupby("Child").Child.count()
test.info()
X_train = train.drop(["Survived", "Name", "Ticket"],axis=1)

Y_train = train["Survived"]

X_test  = test.drop(["PassengerId", "Name", "Ticket"], axis = 1)
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('pySubmission.csv', index=False)