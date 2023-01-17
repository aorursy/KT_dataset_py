# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling



import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

combine = [train_data, test_data]
train_data.head()
train_data.columns
train_data.describe()
train_data.info()
test_data.info()
train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_data[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_data[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_data[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)
g = sns.FacetGrid(train_data, col="Survived")

g.map(plt.hist, "Age", bins=20)
g = sns.FacetGrid(train_data, col="Survived", row="Sex")

g.map(plt.hist, "Fare", bins=20)
sns.catplot(x = "Pclass", y = "Survived", data=train_data, kind = "bar")
sns.catplot(x = "Parch", y = "Survived", data=train_data, kind= "bar")
sns.catplot(x = "Embarked", y = "Survived", data=train_data, kind= "bar", hue = "Sex")
sns.catplot(x = "SibSp", y = "Survived", data=train_data, kind= "bar")
sns.catplot(x = "Sex", y = "Survived", data=train_data, kind= "bar")
## dropping the ticket and cabin feature 

print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_data = train_data.drop(["Ticket", "Cabin"], axis=1)

test_data = test_data.drop(["Ticket", "Cabin"], axis=1)

combine = [train_data, test_data]



"After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape
## extracting the title from the names 

for dataset in combine:

    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=False)



pd.crosstab(train_data["Title"], train_data["Sex"])
## We can replace many titles with a more common name or classify them as Rare.

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_data[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()
## We can now convert the categorical values.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_data.head()
## dropping the name and passenger feature in the train dataset

train_data = train_data.drop(["Name", "PassengerId"], axis=1)

test_data = test_data.drop(["Name"], axis=1)

combine = [train_data, test_data]

train_data.shape, test_data.shape
for dataset in combine:

    dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0} ).astype(int)



train_data.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_data, row="Pclass", col="Sex", size=2.2, aspect=1.6)

grid.map(plt.hist, "Age", alpha=.5, bins=20)

grid.add_legend()
## an empty array to guess age value baseed on pclass and gender combinations 

guess_ages = np.zeros((2,3))

guess_ages
## we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_data = dataset[(dataset["Sex"] == i) & \

                                  (dataset["Pclass"] == j+1)]["Age"].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_data.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    "Age"] = guess_ages[i,j]



    dataset["Age"] = dataset["Age"].astype(int)



train_data.head()
## creating agebands and determinig correlation with survived

train_data["AgeBand"] = pd.cut(train_data["Age"], 5)

train_data[["AgeBand", "Survived"]].groupby(["AgeBand"], as_index=False).mean().sort_values(by="AgeBand", ascending=True)
## replacing age with ordinal based on the bands 

for dataset in combine:    

    dataset.loc[ dataset["Age"] <= 16, "Age"] = 0

    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1

    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2

    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3

    dataset.loc[ dataset["Age"] > 64, "Age"] = 4

train_data.head(10)
train_data = train_data.drop(["AgeBand"], axis=1)

combine = [train_data, test_data]

train_data.head()
## creating a new feature for family size which combines parch and sibsp

for dataset in combine:

    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1



train_data[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean().sort_values(by="Survived", ascending=False)
## creating another feature for isalone

for dataset in combine:

    dataset["IsAlone"] = 0

    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1



train_data[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean()
train_data = train_data.drop(["Parch", "SibSp", "FamilySize"], axis=1)

test_data = test_data.drop(["Parch", "SibSp", "FamilySize"], axis=1)

combine = [train_data, test_data]



train_data.head()
# artificial feature combinig age and pclass together

for dataset in combine:

    dataset["Age*Class"] = dataset.Age * dataset.Pclass



train_data.loc[:, ["Age*Class", "Age", "Pclass"]].head(10)
## filling missing values with S simply its the most occurance

freq_port = train_data.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)

    

train_data[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean().sort_values(by="Survived", ascending=False)
## creating a new feature for embarked

for dataset in combine:

    dataset["Embarked"] = dataset["Embarked"].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_data.head()
test_data["Fare"].fillna(test_data["Fare"].dropna().median(), inplace=True)

test_data.head()
## creating a fareband

train_data["FareBand"] = pd.qcut(train_data["Fare"], 4)

train_data[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean().sort_values(by="FareBand", ascending=True)
## Convert the Fare feature to ordinal values based on the FareBand.



for dataset in combine:

    dataset.loc[ dataset["Fare"] <= 7.91, "Fare"] = 0

    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1

    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"]   = 2

    dataset.loc[ dataset["Fare"] > 31, "Fare"] = 3

    dataset["Fare"] = dataset["Fare"].astype(int)



train_data = train_data.drop(["FareBand"], axis=1)

combine = [train_data, test_data]

    

train_data.head(10)
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
## Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_data = pd.DataFrame(train_data.columns.delete(0))

coeff_data.columns = ["Feature"]

coeff_data["Correlation"] = pd.Series(logreg.coef_[0])



coeff_data.sort_values(by="Correlation", ascending=False)
## Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
## KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
## Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
## Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
## Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
## Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
## Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
## Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    "Model": ["Support Vector Machines", "KNN", "Logistic Regression", 

              "Random Forest", "Naive Bayes", "Perceptron", 

              "Stochastic Gradient Decent", "Linear SVC", 

              "Decision Tree"],

    "Score": [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by="Score", ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred})

submission.to_csv("Titanic_Pred.csv", index=False)
submission.head()