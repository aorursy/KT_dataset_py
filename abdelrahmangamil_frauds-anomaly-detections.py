# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns 
data = pd.read_csv('../input/creditcard.csv')

data.info()
sns.countplot(data["Class"])
fraud_ratio = float(data["Class"][data["Class"] == 1].shape[0])/data.shape[0] 
data.head()
data["Class"].value_counts()
from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM
X =  data.iloc[:,:-1]

Y = data["Class"]
from sklearn.model_selection import train_test_split

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y.shape,X.shape
classifiers = {

    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X))

    

   

}
n_outliers = 492

for i, (clf_name,clf) in enumerate(classifiers.items()):

    #Fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        y_pred = clf.fit_predict(X)

        scores_prediction = clf.negative_outlier_factor_

    elif clf_name == "Support Vector Machine":

        clf.fit(X)

        y_pred = clf.predict(X)

    else:    

        clf.fit(X)

        scores_prediction = clf.decision_function(X)

        y_pred = clf.predict(X_test)

    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y_test).sum()

    # Run Classification Metrics

    print("{}: {}".format(clf_name,n_errors))

    print("Accuracy Score :")

    print(accuracy_score(Y_test,y_pred))

    print("Classification Report :")

    print(classification_report(Y_test,y_pred))