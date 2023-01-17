# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame

from pandas.tools.plotting import scatter_matrix



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get train & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview the data

titanic_df.head()
# information about the dataframe

titanic_df.info(verbose=True)

print("----------------------------")

test_df.info()
titanic_df = titanic_df.drop(['PassengerId'], axis=1)



titanic_df = titanic_df.dropna(subset=['Embarked'])

titanic_df.info() # verify
# Get some statistical values for the training data set.

titanic_df.describe()
# Get some statistical values for the training data set.

test_df.describe()
print(titanic_df.corr())
# Overall histogram summary

titanic_df.hist(figsize=(12,12), layout=(4,2));
scatter_matrix(titanic_df, alpha=0.2, figsize=(12, 12), diagonal='kde');
# Survival

plt.figure(figsize=(6,6))

titanic_df.Survived.value_counts().plot(kind='bar', color="blue", alpha=.65)

plt.title("Survival Breakdown (1 = Survived, 0 = Died)");
# PClass

fig = plt.figure(figsize=(12,6))



ax1 = plt.subplot2grid((1,3), (0,0), colspan=2)

# sub_plot1 = fig.add_subplot(111)

titanic_df.Pclass.value_counts().plot(kind='bar', color="blue", alpha=.65)

plt.title("Ticket Class")



# sub_plot1 = fig.add_subplot(111)

ax2 = plt.subplot2grid((1,3), (0,2))

df = pd.DataFrame([

    titanic_df[titanic_df['Survived']==1]['Pclass'].value_counts(),

    titanic_df[titanic_df['Survived']==0]['Pclass'].value_counts()])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, ax=ax2)



sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5);
# TODO: Visualise and Analyse

# PClass

# Name

# Sex

# Age

# SibSp

# Parch

# Ticket

# Fare

# Cabin

# Embarked
# TODO:
# drop string values - these should be convirted

titanic_df = titanic_df.drop(["Name", "Ticket", "Cabin", "Embarked"],axis=1)

titanic_df = titanic_df.dropna()

test_df = test_df.drop(["Name", "Ticket", "Cabin", "Embarked"],axis=1)

test_df = test_df.dropna()



# Replace empty age values with the mean - should verify mean / median. Perhaps take by analysing

# name to determin title e.g. Mrs. is probably older than Miss.

titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

test_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)



# mapping string values to numerical one 

titanic_df['Sex'] = titanic_df['Sex'].map({'male':1,'female':0})

test_df['Sex'] = test_df['Sex'].map({'male':1,'female':0})



# convert from float to int

titanic_df['Age'] = titanic_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)
# define training and testing sets



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_predLogisticRegression = logreg.predict(X_test)



print(logreg.score(X_train, Y_train))



scores = cross_validation.cross_val_score(logreg, X_train, Y_train, cv=5)

print(scores)



print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



Y_predSVC = svc.predict(X_test)



svc.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_predRandomForest = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
# K Nearest Neighbors



knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, Y_train)



Y_pred = knn.predict(X_test)



knn.score(X_train, Y_train)
# Gaussian Naive Bayes



gaussian = GaussianNB()



gaussian.fit(X_train, Y_train)



Y_predGaussian = gaussian.predict(X_test)



gaussian.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_predRandomForest

    })

submission.to_csv('titanic.csv', index=False)