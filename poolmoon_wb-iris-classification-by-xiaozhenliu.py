# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.



# Read the input file and put the data to pandas' dataframe format.

df = pd.read_csv("../input/Iris.csv")
df.head()
df.ix[:,-1].unique()
df.drop('Id', axis=1, inplace=True)
# TODO: Set up for multiple classifier!

# Hint: use 0, 1, 2 to represent three different types of iris.

df = df.replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
# TODO: Prepare training/testing data

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.4, random_state=0)

X_train = train.ix[:,:-1]

y_train = train.ix[:,-1]

X_test = test.ix[:,:-1]

y_test = test.ix[:,-1]
# TODO: create a classifier



from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

clf = OneVsRestClassifier(LinearSVC(random_state=0))
# TODO: make predictions

clf.fit(X_train, y_train) # Train the learner

pred_train = clf.predict(X_train) # Predictions on the training set

pred_test = clf.predict(X_test) #The predictions on the test set
# TODO: Choose your own performance measures and calculate them

from sklearn import metrics



def print_results(y_train, y_test, pred_train, pred_test):

    accuracy_train = metrics.accuracy_score(y_train, pred_train)

    accuracy_test = metrics.accuracy_score(y_test, pred_test)

    print("train accuracy = {}, test accuracy = {}".format(accuracy_train, accuracy_test))

    F1_micro_test = metrics.f1_score(y_test, pred_test, average='micro')

    F1_macro_test = metrics.f1_score(y_test, pred_test, average='macro')

    print("F1 micro for test = {}, F1 macro for test = {}".format(F1_micro_test, F1_macro_test))

    

print_results(y_train, y_test, pred_train, pred_test)

# create a classifier using one vs one



from sklearn.multiclass import OneVsOneClassifier

from sklearn.svm import LinearSVC

clf = OneVsOneClassifier(LinearSVC(random_state=0))



# make predictions

clf.fit(X_train, y_train) # Train the learner

pred_train = clf.predict(X_train) # Predictions on the training set

pred_test = clf.predict(X_test) #The predictions on the test set
print_results(y_train, y_test, pred_train, pred_test)