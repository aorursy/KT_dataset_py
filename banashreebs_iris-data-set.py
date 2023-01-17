# Importing useful libraries.

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import shapiro

import seaborn as sns





# Load dataset

dataset = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')



# Checking for null and categorical variables.

dataset.isnull().sum()

dataset.eq(0).sum()
dataset.info()
dataset.head()
sns.boxplot(data = dataset, orient = 'h')

plt.show()
sns.pairplot(dataset, hue = 'species')

plt.show()

X = dataset.iloc[:,:-1]

Y = dataset.iloc[:,-1]



# Converting categorical to num

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

Y = lb.fit_transform(Y)



#splitting into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



#Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)



#Accuracy

from sklearn.metrics import accuracy_score

print('Accuracy score(LR) = ',accuracy_score(Y_test, Y_pred))



#Report

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

from sklearn.metrics import classification_report

print('Classificaation Report(LR) = \n',classification_report(Y_test, Y_pred, target_names  = target_names))
