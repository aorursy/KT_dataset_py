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
%matplotlib inline



from pandas import DataFrame, Series

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import statsmodels.api as sm

from sklearn.cross_validation import train_test_split



import seaborn as sns

from sklearn.model_selection import train_test_split



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



X = train.select_dtypes(include=['int64', 'float64']).drop(['SalePrice', 'Id'], axis=1)

Y = train['SalePrice']

# Count missing values in training data set

#print(pd.isnull(X).sum())



X = X.fillna(X.mean())

Y = Y.fillna(Y.mean())

#print(pd.isnull(X).sum())

#print(X)

#print(Y)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)





from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    print(train_predictions)

    #acc = accuracy_score(y_test, train_predictions)

    #print("Accuracy: {:.4%}".format(acc))

    

    #train_predictions = clf.predict_proba(X_test)

    #ll = log_loss(y_test, train_predictions)

    #print("Log Loss: {}".format(ll))

    

    #log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    #log = log.append(log_entry)

    

print("="*30)