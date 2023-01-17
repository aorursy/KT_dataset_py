import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')

import seaborn as sns



# ML imports

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris_df = pd.read_csv('../input/iris/Iris.csv', index_col = 'Id')

iris_df.head()
sns.pairplot(iris_df, hue = 'Species', height = 3.5)

plt.show()
iris_df.hist(figsize = (12,8))

plt.show()
fig, ax = plt.subplots(figsize = (12, 8))

sns.heatmap(iris_df.corr(), annot=True, cmap = 'RdGy')

plt.show()
# A function for training the different classifiers on the data

def classifiers(X_train, X_test, y_train, y_test):



    # Model names

    names = [

        'Logistic Regression:',

        'Naive Bayes:',

        'KNN:',

        'Decision Tree:',

        'Random Forest:',

        'Support Vector Machine:'

    ]



    # Instantiating the estimators

    clfs = [

        LogisticRegression(solver = 'liblinear', multi_class = 'auto'),

        GaussianNB(),

        KNeighborsClassifier(),

        DecisionTreeClassifier(),

        RandomForestClassifier(n_estimators = 10, random_state = 1002),

        SVC(kernel = 'linear',gamma = 'auto')

    ]

    print('Accuracies:')



    # Building a model with each classifier and evaluating the accuracy

    for name, clf in zip(names, clfs):

        mdl = clf

        mdl = mdl.fit(X_train, y_train)

        preds = mdl.predict(X_test)

        print(name, '{:.4f}'.format(accuracy_score(y_test, preds)))
# First we split the data into training and testing sets in order to evaluate the classifiers

X_train, X_test, y_train, y_test = train_test_split(iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], iris_df['Species'],

                                                   random_state = 1002)

classifiers(X_train, X_test, y_train, y_test)
# Splitting into training and testing sets but using 3 features this time.

X_train, X_test, y_train, y_test = train_test_split(iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']], iris_df['Species'],

                                                   random_state = 1002)



classifiers(X_train, X_test, y_train, y_test)
splitter = StratifiedKFold(n_splits = 5, random_state = 1002)



X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

Y = iris_df['Species']



# Model names

names = [

    'Logistic Regression:',

    'Naive Bayes:',

    'KNN:',

    'Decision Tree:',

    'Random Forest:',

    'Support Vector Machine:'

]



# Instantiating the estimators

clfs = [

    LogisticRegression(solver = 'liblinear', multi_class = 'auto'),

    GaussianNB(),

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 10, random_state = 1002),

    SVC(kernel = 'linear',gamma = 'auto')

]

print('Accuracies using 5-fold cross validation:')



for name, clf in zip(names, clfs): # Iterating through classifiers

    acc = [] # List to store accuracy for each fold

    # Iterating through training and testing sets in each fold

    for train_idx, test_idx in splitter.split(X, Y):

        mdl = clf.fit(X.iloc[train_idx], Y.iloc[train_idx]) # Fitting model

        preds = mdl.predict(X.iloc[test_idx]) # Making predictions

        acc.append(accuracy_score(Y.iloc[test_idx], preds))

    print(name, '{:.4f}'.format(np.mean(acc)), '±',  '{:.4f}'.format(np.std(acc))) # Calculating the mean and standard deviation of each model