import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input/heart-disease-uci"))
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.target.value_counts()
countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
y = df.target.values

x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices

x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
from sklearn.svm import SVC
accuracies = {}



svm = SVC(random_state = 1)

svm.fit(x_train.T, y_train.T)



acc = svm.score(x_test.T,y_test.T)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)



acc = nb.score(x_test.T,y_test.T)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))