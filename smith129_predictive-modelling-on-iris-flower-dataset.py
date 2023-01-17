# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import scipy

import matplotlib

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#matplotlib for plotting various graphs

#sklearn for accessing different algorithms



import pandas

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import matplotlib.pyplot as plt 



#Load data from the Iris Dataset as a Pandas Dataframe into 'data'

data = pd.read_csv("../input/Iris.csv")



#Dimensions of the dataset

print("Dimensions =",data.shape)         

print('\nIris Data\n')

print(data.head(10))      #Displaying the First 10 rows of the Iris Dataset



#Removing the first column as it is redundant

del data['Id']



#Renaming the column names with something more accesible

data.columns=['Sepal-Length', 'Sepal-Width', 'Petal-Length', 'Petal-Width', 'Class']

print(data.head(5))

print('\n')



#Class Distribution of the Dataset

print("Class Distribution\n")

print(data.groupby('Class').size())  



#Statistical Summary of the Dataset

print("\n\nStatistical Summary\n")

print(data.describe())               

print('\n')

#Histogram



data.hist(edgecolor='black', linewidth=1.2)

plt.show()



#Box and Whisper Plots

data.plot(kind='box',subplots=True,layout=(2,2), sharex=False,sharey=False,title="Boxplot(Class vs cm)")

plt.show()



#Multivariate Plot 

data.plot(kind="scatter", x="Sepal-Length", y="Sepal-Width")

plt.title("Sepal Length(cm) vs Sepal Width(cm)")

plt.show()



data.plot(kind="scatter", x="Petal-Length", y="Petal-Width")

plt.title("Petal Length(cm) vs Petal Width(cm)")

plt.show()



#Create a Validation Dataset by splitting the original dataset into separate Test and Train datasets

array=data.values

X = array[:,0:4]

Y = array[:,4]

print("Training set for both X and Y is about 80% of the original dataset")

validation_size = 0.20

print("Validation Size=",validation_size)

seed = 7 #Setting up randomness

#With model_selection.train_test_split imported from sklearn, we can split data into test and train sets

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print(X_train[5])

#Use K Fold cross validation to estimate accuracy, k =10

#Meaning that the dataset will be split into 10 parts, which will train on 9 and test on 1 for every combination

seed=7

scoring='accuracy'

#We will be using the scoring variable to compare the accuracy of every model tested



print("Evaluate every Algorithm\n")

# pot Check Different Algorithms

#Linear(LR, LDA) & Non Linear (KNN, CART, NB, SVM)

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

#Evaluate each model in turn

results = []

names = []

print("Model\t","Mean\t\t","Std")

for name, model in models: 

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s:\t %f\t (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)







#From the above values, we can see that Support Vector Machines (SVM) has the largest estimated accuracy score.

#Compare mean acccuracy of Algorithms with each other

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.svm import SVC

model=SVC()

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)

print("Accuracy=",accuracy_score(Y_validation, predictions),"%\n")

print("Confusion Matrix=\n",confusion_matrix(Y_validation, predictions),"\n")

print("Classification Report=\n",classification_report(Y_validation, predictions))