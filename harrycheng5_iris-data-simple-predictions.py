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
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set();

%matplotlib inline





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

sns.scatterplot(df.sepal_length, df.sepal_width, hue=df.species, ax=ax1);

sns.scatterplot(df.petal_length, df.petal_width, hue=df.species, ax=ax2);
from sklearn.preprocessing import MinMaxScaler, LabelEncoder



X = df.iloc[:, :-1].values

y = df.iloc[:,-1].values



# scale each feature between 0 and 1

scaler = MinMaxScaler().fit(X)

scaled_X = scaler.transform(X)



# label target

le = LabelEncoder()

labeled_y = le.fit_transform(y)





print('New X')

print(scaled_X[:5])

print('New y')

print(labeled_y[:5])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# split the data

X_train, X_test, y_train, y_test = train_test_split(scaled_X, labeled_y, stratify=labeled_y, random_state=0)



# use default for all of them to compare

classifiers = {'Logistic': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(), 

               'SVM': SVC(), 'K Neighbors': KNeighborsClassifier()}



score_df = pd.DataFrame(index=['training set', 'test set'])

for clf in classifiers:

    classifiers[clf].fit(X_train, y_train)

    train_score = round(accuracy_score(y_train, classifiers[clf].predict(X_train)),4)

    test_score = round(accuracy_score(y_test, classifiers[clf].predict(X_test)),4)

    score_df[clf] = [train_score, test_score]

    

score_df