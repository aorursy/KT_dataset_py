# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Numpy and Pandas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Machine Learning

from patsy import dmatrices

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_selection import RFE

from sklearn import metrics



# graphical modules

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
# import test and train and combine to full data set 

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



titanic = train.append(test, ignore_index = True)



# printing the first row and structure

print("The number of records are as follows:")

print("Training set: {}".format(train["Age"].count()))

print("Test set: {}".format(test["Age"].count()))

print("\nBelow see an example record \n")

print(titanic.iloc[1])

print("\nBelow see the data type for each column/variable \n")

print(titanic.dtypes)
# Return missing values in dataframe for training set

print("Total Records: {}\n".format(titanic["Age"].count()))



print("Training set missing values")

print(train.isnull().sum())

print("\n")



print("Testing set missing values")

print(test.isnull().sum())
# replace each Null Age with the mean

train["Age"].fillna(train["Age"].mean(), inplace=True)



# apply the same preprocessing to test set

test["Age"].fillna(test["Age"].mean(), inplace=True)

test["Fare"].fillna(test["Fare"].mean(), inplace=True)



# confirm values are no longer null

print("Training set missing values after pre-processing")

print(train.isnull().sum())

print("\n")
# Seaborn Violin Plot would be a nice way to look at this relationship

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train,

               split=True, cut =0, inner="stick", palette="Set1")
# swarm plot of embarked and fare

sns.swarmplot(x="Pclass", y="Fare", hue="Survived", data=train, palette="dark")
# create Child column into dataframe for train and test using a list comprehension

train["Child"] = ["Child" if int(x) < 18 else "Adult" for x in train["Age"]]

test["Child"] = ["Child" if int(x) < 18 else "Adult" for x in test["Age"]]
# Family size column

train["Family_size"] = train["Parch"].astype(np.int64) + train["SibSp"].astype(np.int64) + 1 

test["Family_size"] = test["Parch"].astype(np.int64) + test["SibSp"].astype(np.int64) + 1 



print("See below a record for a child with sibling and parent, we know have a Child and Family Size indicator: \n")

print(train.iloc[10])
# Produce heatmap

family = train.pivot_table(values="Survived", index = ["Child"], columns = "Family_size")



# Draw a heatmap with the numeric values in each cell

htmp = sns.heatmap(family, annot=True, cmap="YlGn")
# Use regex and str.extract method to extract title from name for test and train

train["Title"] = train["Name"].str.extract("\,\s(.*?)\." , expand=True)

train["Title"].str.strip(" ")

test["Title"] = test["Name"].str.extract("\,\s(.*?)\." , expand=True)

test["Title"].str.strip(" ")



# Print list of values and the count for that data frame series

train["Title"].value_counts(ascending = False)
# roll-up titles

train["Title"] = [x if x in ["Miss", "Mr", "Mrs", "Master", "Dr", "Rev"] else "Vip" for x in train["Title"] ]

test["Title"] = [x if x in ["Miss", "Mr", "Mrs", "Master", "Dr", "Rev"] else "Vip" for x in test["Title"] ]



# Seaborn Plot to show survival based on Title

bar = sns.barplot("Title", "Survived", data = train, palette="Greys")

bar.set_ylabel("Chance of Survival")
# create mothers

def mother(row):

  if row["Child"] == "Adult" and row["Sex"] == "female" and row["Title"] == "Mrs" and row["Parch"] > 0:

    return "Mother"

  else:

    return "Not Mother"



train["Mother"] = train.apply(mother, axis=1)

test["Mother"] = test.apply(mother, axis=1)



print("See below and example record for who we believe to be a mother:\n")

print(train.iloc[25])
# format using patsy to get a matrix to pass into the LR model

y, X = dmatrices('Survived ~ Title + Age + Sex + Child + Family_size + Mother + Fare',

                  train, return_type="dataframe")

# flatten y into a 1-D array

y = np.ravel(y)



# evaluate the model by splitting into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# instantiate a logistic regression model, and fit with X and y

LRmodel = LogisticRegression()

LRmodel = LRmodel.fit(X_train, y_train)



# check the accuracy on the training set

LRmodel.score(X_train, y_train)
# what percentage had affairs?

y.mean()
# check the accuracy on the test set

predicted = LRmodel.predict(X_test)

print(metrics.accuracy_score(y_test, predicted))
# evaluate the model using 10-fold cross-validation

scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)



print("List of Scores for CV Folds:")

[print(score) for score in scores]



print("\nMean Accuracy")

print(scores.mean())
# add Survived column to test dataframe (blank)

test["Survived"] = ""



# format using patsy to get a matrix to pass into the LR model

yt, Xt = dmatrices('Survived ~ Title + Age + Sex + Child + Family_size + Mother + Fare',

                  test, return_type="dataframe")

# flatten y into a 1-D array

yt = np.ravel(yt)



# predict on the data

test["Survived"] = LRmodel.predict(Xt)



print("Chance of Survival for Passengers in Test Data:")

print(test["Survived"].mean())
# instantiate Random Forest Classifier and train

RFCmodel = RandomForestClassifier(n_estimators =1000)

RFCmodel = RFCmodel.fit(X_train, y_train)



# print score for training set

print("Train data accuracy:")

print(RFCmodel.score(X_train, y_train))



# check the accuracy on the test set

predicted = RFCmodel.predict(X_test)

print("\nTest data accuracy:")

print(metrics.accuracy_score(y_test, predicted))
leafsizes = []



for x in range(1,110,2):

    RFCmodel = RandomForestClassifier(n_estimators =100, min_samples_leaf= x)

    

    RFCmodel = RFCmodel.fit(X_train, y_train)

    RFC_acc = (RFCmodel.score(X_train, y_train))

    

    predicted = RFCmodel.predict(X_test)

    RFC_test_acc = metrics.accuracy_score(y_test, predicted)

    

    diff_mag = (((RFC_acc - RFC_test_acc)**2)**0.5)

    

    leafsizes.append(("leaf {0}".format(x), RFC_acc, RFC_test_acc, diff_mag))



leafsizes = list(reversed(sorted(leafsizes, key=lambda tup: tup[2])))

    

for i in range(5):

    print(leafsizes[i])
# assign RFC model with the chosen  min_samples_leaf

RFCmodel = RandomForestClassifier(n_estimators =100, min_samples_leaf= 15)

RFCmodel = RFCmodel.fit(X_train, y_train)



# evaluate the model using 10-fold cross-validation

scores = cross_val_score(RandomForestClassifier(n_estimators =100, min_samples_leaf= 15), X, y, scoring='accuracy', cv=10)



print("List of Scores for CV Folds:")

[print(score) for score in scores]



print("\nMean Accuracy")

print(scores.mean())



print(RFCmodel.feature_importances_)
# run model and place values in test dataframe

test["Survived"] = LRmodel.predict(Xt)



# produce submission format

submission_lr = pd.DataFrame()



submission_lr["PassengerId"] = test["PassengerId"]

submission_lr["Survived"] = test["Survived"]



print("Check format:\n")

print(submission_lr.head())



submission_lr.to_csv("Submission_lr.csv", index = False)
# run model and place values in test dataframe

test["Survived"] = RFCmodel.predict(Xt)



# produce submission format

submission_rfc = pd.DataFrame()



submission_rfc["PassengerId"] = test["PassengerId"]

submission_rfc["Survived"] = test["Survived"]



print("Check format:\n")

print(submission_rfc.head())



submission_rfc.to_csv("Submission_rfc.csv", index = False)