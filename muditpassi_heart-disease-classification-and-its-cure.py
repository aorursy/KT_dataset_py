# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the csv file

data = pd.read_csv("../input/heart.csv")

data.head()
#check the null values in data and find its counts

data.isnull().sum()
# the data contains no null value. Now we will count the target values

data['target'].value_counts()

# Data visualization

sns.countplot(data.target)
pd.crosstab(data['age'], data['target']).plot(kind='bar', figsize=(20, 8))

plt.title('Heart Disease frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.legend(["Don't have disease", "Have disease"])

plt.savefig('heartDiseaseForAge.png')

plt.show()
pd.crosstab(data['sex'], data['target']).plot(kind='bar', figsize=(15, 6))

plt.title('Heart Disease frequency for Sex')

plt.xlabel('Sex(Female-0, Male-1)')

plt.xticks(rotation=0)

plt.ylabel('Frequency')

plt.legend(["Don't have disease", "Have disease"])

plt.show()
# Now seperate the data as response variable and feature variable



X_data = data.drop('target', axis=1)

y = data['target']



# Normalisation



X = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data)).values
# Train and Test splitting of data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)
# Applying Standard Scaling to get optimized results



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

X_train
# Random forest Classifier



rand_fc = RandomForestClassifier(n_estimators=200)

rand_fc.fit(X_train,y_train)

pred_rfc = rand_fc.predict(X_test)

print(classification_report(y_test, pred_rfc))

print('Confusion Matrix:',confusion_matrix(y_test, pred_rfc))

print('Accuracy:',accuracy_score(y_test, pred_rfc))
# Support Vector Classifier (SVC)



classifier = SVC()

classifier.fit(X_train, y_train)

pred_classi = classifier.predict(X_test)

print(classification_report(y_test, pred_classi))

print('Confusion Matrix:',confusion_matrix(y_test, pred_classi))

print('Accuracy:',accuracy_score(y_test, pred_classi))
# MLP Classifier

mlpc = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500)

mlpc.fit(X_train, y_train)

pred_mlpc = mlpc.predict(X_test)

print(classification_report(y_test, pred_mlpc))

print('Confusion Matrix:',confusion_matrix(y_test, pred_mlpc))

print('Accuracy:',accuracy_score(y_test, pred_mlpc))