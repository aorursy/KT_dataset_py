# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import the data

diabetes = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
#gets name of column

diabetes.columns
#information about the column

diabetes.info()
#get basic statistics about the data

diabetes.describe()
# check for the null values

diabetes.isnull().sum()
#let us look at first 10 rows

diabetes.head(10)
def check_for_zero(columns):

    for col in columns:

        if 0 in diabetes[col]:

            print(col+' has 0 in it.')



columns = ['Glucose', 'BloodPressure', 'SkinThickness',

           'BMI', 'DiabetesPedigreeFunction', 'Age',]

check_for_zero(columns)
#before doing so, we will split our X and y

X = diabetes.drop('Outcome',axis=1)

Y = diabetes['Outcome']
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=0, strategy='mean')

X[columns] = imp.fit_transform(X[columns]) #fit the imputer
X.head(10)
X = X.drop('Insulin',axis=1)
#let's do some EDA

import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(X['Pregnancies'])

X['Pregnancies'].value_counts()
def draw_dist(column):

    plt.figure()

    return sns.distplot(X[col])
for col in columns:

    draw_dist(col)
#Let us have some visualization about y

sns.countplot(Y)
#since all columns are not on same scale so let us normalize them

# Import the necessary modules

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

# Setup the pipeline steps: steps

steps = [('scaler', StandardScaler()),

        ('knn', KNeighborsClassifier(n_neighbors=5))]

        

# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)



# Fit the pipeline to the training set: knn_scaled

knn_scaled = pipeline.fit(X_train,y_train)



# Instantiate and fit a k-NN classifier to the unscaled data

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)



# Compute and print metrics

print('Accuracy on training data: {}'.format(knn_scaled.score(X_train,y_train)))

print('Accuracy on test data: {}'.format(knn_scaled.score(X_test,y_test)))

#implement logistic regression

# Setup the pipeline steps: steps

steps = [('scaler', StandardScaler()),

        ('knn', LogisticRegression())]

        

# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)



# Fit the pipeline to the training set: knn_scaled

logreg_scaled = pipeline.fit(X_train,y_train)



# Compute and print metrics

print('Accuracy on training data: {}'.format(logreg_scaled.score(X_train,y_train)))

print('Accuracy on test data: {}'.format(logreg_scaled.score(X_test,y_test)))

#implement logistic regression

# Setup the pipeline steps: steps

steps = [('scaler', StandardScaler()),

        ('knn', SVC())]

        

# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.4, random_state=42)



# Fit the pipeline to the training set: knn_scaled

Svm_scaled = pipeline.fit(X_train,y_train)



# Compute and print metrics

print('Accuracy on training data: {}'.format(Svm_scaled.score(X_train,y_train)))

print('Accuracy on test data: {}'.format(Svm_scaled.score(X_test,y_test)))
