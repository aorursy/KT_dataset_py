# Import libraries used



import pandas as pd

from pandas import read_csv

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
# Import files used for Train and Test

ds = read_csv("../input/titanic/train.csv")

tds = read_csv("../input/titanic/test.csv")
# Check the data in Train dataset

ds.head(3)
# Check the column data types

ds.dtypes
ds.shape
tds.shape
# Describe the content (data) of Train set

ds.describe()
# Check if there are any Nulls in Train set

ds.isnull().sum()
# Check if there are any Nulls in Test set

tds.isnull().sum()
# Remove the columns Name, Cabin and Ticket from both Train and Test sets

del ds['Name']

del ds['Cabin']

del ds['Ticket']



del tds['Name']

del tds['Cabin']

del tds['Ticket']
# for the Nulls in Train and Test datasets, use the mean to replace the Nulls



ds.Age = ds.Age.fillna(ds.Age.mean())

tds.Age = tds.Age.fillna(tds.Age.mean())

tds.Fare = tds.Fare.fillna(tds.Fare.mean())
# Identify the unique values for Gender 

gender_values = ds.drop_duplicates('Sex')

gender_values.Sex
# Identify the unique values for Embarked

Embarked_values = ds.drop_duplicates('Embarked')

Embarked_values.Embarked
# Replace the Null values in Embarked and Gender with the mode of each column

ds.Embarked = ds.Embarked.fillna(ds.Embarked.mode()[0])

tds.Embarked = tds.Embarked.fillna(tds.Embarked.mode()[0])
# Convert labels for Gender and Embarked to numbers

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



le.fit(ds.Sex)

ds.Sex = le.transform(ds.Sex)

le.fit(tds.Sex)

tds.Sex = le.transform(tds.Sex)



le.fit(ds.Embarked)

ds.Embarked = le.transform(ds.Embarked)

le.fit(tds.Embarked)

tds.Embarked = le.transform(tds.Embarked)
# Check the data after transformations

ds.head(5)
tds.head(5)
# Plot histogram for the data

ds.hist()

plt.show()
# Check the correlation for Train set

correlations = ds.corr()

print (correlations)
# Plot correlations

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, interpolation='nearest', vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 10, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(ds.columns)

ax.set_yticklabels(ds.columns)

plt.show()
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler



#ColTrain = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']

ColTrain = ['PassengerId', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked', 'Survived']

MLTrain = ds[ColTrain]

arrayTrain = MLTrain.values



#ColTest = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

ColTest = ['PassengerId', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']

MLTest = tds[ColTest]

arrayTest = MLTest.values



#X_Train = arrayTrain[:,0:8]

#Y_Train = arrayTrain[:,8]

#X_Test = arrayTest[:,0:8]



X_Train = arrayTrain[:,0:7]

Y_Train = arrayTrain[:,7]

X_Test = arrayTest[:,0:7]



#X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)



scaler = MinMaxScaler(feature_range = (0, 1))

rescaledXTrain = scaler.fit_transform(X_Train)

rescaledXTest = scaler.fit_transform(X_Test)
# Run Models and compare



models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, rescaledXTrain, Y_Train, cv=kfold, scoring='accuracy')

    #cv_results = cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
pyplot.boxplot(results, labels=names)

pyplot.title('Algorithm Comparison')

pyplot.show()
# Run Logistic Regression

lr = LogisticRegression()



lr.fit(rescaledXTrain, Y_Train)

lr.score(rescaledXTrain, Y_Train)



print('Coefficient: \n', lr.coef_)

print('Intercept: \n', lr.intercept_)



prediction_lr = lr.predict(rescaledXTest)



prediction_lr
accuracy_log = round(lr.score(rescaledXTrain, Y_Train)*100,2)

accuracy_log
# Generate Submission File

survivedLR = prediction_lr.astype(int)



submissionLR = pd.DataFrame({

    "PassengerId": tds["PassengerId"],

    "Survived": survivedLR

})



submissionLR.to_csv('submissionlr.csv', index=False)