# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

import seaborn as sns

from sklearn import neighbors, model_selection,metrics

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
inputPath = "/kaggle/input/prostate-cancer/Prostate_Cancer.csv"

df = pd.read_csv(inputPath)

#Deleting the unique column

del df['id']

#First 10 Records

df.head(10)
# 2. What types of crimes are most common?

OFFENSE_CODE_GROUP = df['diagnosis_result'].value_counts(sort = True)

plt.figure(figsize=(20,10))

sns.barplot(OFFENSE_CODE_GROUP.index, OFFENSE_CODE_GROUP.values, alpha=0.8)

plt.title('Diagnosis Result of The People who Have Cancer', fontsize=30)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Diagnosis Result', fontsize=12)

plt.show()
# 2. What types of crimes are most common?

OFFENSE_CODE_GROUP = df['perimeter'].value_counts(sort = True)

plt.figure(figsize=(20,10))

sns.barplot(OFFENSE_CODE_GROUP.index, OFFENSE_CODE_GROUP.values, alpha=0.8)

plt.title('Number of Occurrences vs Number of Occurence', fontsize=30)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Perimeter', fontsize=12)

plt.show()
#Making and Splitting The Training and Test Data

df['compactness'].fillna(df['compactness'].mean(),inplace=True)

df['fractal_dimension'].fillna(df['fractal_dimension'].mean(),inplace=True)

X = np.c_[df['radius'], df['texture'], df['perimeter'], df['area'], df['smoothness'], df['compactness'], df['symmetry'], df['fractal_dimension']]

y = df['diagnosis_result']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0, stratify = y)

#Using KNN For Prediction

n_neighbors = 15

kNeighborsClassifier = neighbors.KNeighborsClassifier(n_neighbors)

kNeighborsClassifier.fit(X_train, y_train)

cancerPrediction = kNeighborsClassifier.predict(X_test)

# How did our model perform?

count_misclassified = (y_test != cancerPrediction).sum()

print('kNeighborsClassifier Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, cancerPrediction)

print('kNeighborsClassifier Accuracy: {:.2f}'.format(accuracy))
#Using Decision Tree For Prediction

decisionTreeClassifier = tree.DecisionTreeClassifier()

decisionTreeClassifier.fit(X_train, y_train)

cancerPrediction = decisionTreeClassifier.predict(X_test)

# How did our model perform?

count_misclassified = (y_test != cancerPrediction).sum()

print('DecisionTreeClassifier Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, cancerPrediction)

print('DecisionTreeClassifier Accuracy: {:.2f}'.format(accuracy))
#Using Gauian Naive Bayes For Prediction

naiveBayesClassifier = GaussianNB()

naiveBayesClassifier.fit(X_train, y_train)

cancerPrediction = naiveBayesClassifier.predict(X_test)

# How did our model perform?

count_misclassified = (y_test != cancerPrediction).sum()

print('NaiveBayesClassifier Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, cancerPrediction)

print('NaiveBayesClassifier Accuracy: {:.2f}'.format(accuracy))