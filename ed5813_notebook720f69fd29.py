# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import f1_score

from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV

from sklearn import svm

from time import time
def train_predict(clf, X_train, y_train, X_test, y_test):

    ''' Train and predict using a classifer based on F1 score. '''

    

    # Indicate the classifier and the training set size

    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    

    # Train the classifier

    train_classifier(clf, X_train, y_train)

    

    # Print the results of prediction for both training and testing

    train_score = predict_labels(clf, X_train, y_train)

    test_score = predict_labels(clf, X_train, y_train)

    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))



    test_score = predict_labels(clf, X_test, y_test)

    print ("F1 score for test set: {:.4f}.".format(test_score))

    return test_score
def train_classifier(clf, X_train, y_train):

    ''' Fits a classifier to the training data. '''

    

    # Start the clock, train the classifier, then stop the clock

    start = time()

    clf.fit(X_train, y_train)

    end = time()

    

    # Print the results

    diff = end-start

    print ("Trained model in {:.4f} seconds".format(diff))
def predict_labels(clf, features, target):

    ''' Makes predictions using a fit classifier based on F1 score. '''

    

    # Start the clock, make predictions, then stop the clock

    start = time()

    y_pred = clf.predict(features)

    end = time()

    

    # Print and return results

    diff = end-start

    print ("Made predictions in {:.4f} seconds.".format(diff))

    return f1_score(target.values, y_pred, pos_label=1, average='weighted')
def add_pred_cols(clf, df):

    feature_cols, target_col, X_all, y_all = create_analysis_vars(df)

    y_pred = clf.predict(df[feature_cols])

    pred = [int(y) for y in y_pred]

    return pred
def create_analysis_vars(df):

    feature_cols = list(df.columns[1:])

    target_col = df.columns[0]

    X_all = df[feature_cols]

    y_all = df[target_col]

    return (feature_cols, target_col, X_all, y_all)
data = pd.read_csv(r'../input/train.csv')

#test = pd.read_csv(r'../input/test.csv')



clf = svm.SVC(random_state=0)



feature_cols, target_col, X_all, y_all = create_analysis_vars(data)

#normalise

X_all = X_all.asty

X_all /= 255

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all)

score = train_predict(clf, X_train, y_train, X_test, y_test)

print("Score: {:.4f}".format(score))







