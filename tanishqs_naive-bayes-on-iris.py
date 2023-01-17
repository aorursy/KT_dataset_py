# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename = "/kaggle/input/iris/Iris.csv"
data = pd.read_csv(filename)
# print(data.head())

y = data.Species
X = data.drop('Species', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
# print("\nX_train:\n")
# print(X_train.head())
# print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

gnb = GaussianNB()

#Trained the model using the training sets
gnb.fit(X_train, y_train)

#Predicted the response for test dataset
y_pred = gnb.predict(X_test)
y_pred_series = pd.Series(y_pred)


# print 'Assignment \t\tGrade' 
# for xtest, ytest, ypred in zip(X_test, y_test, y_pred_series):
#     print (xtest + '\t' + ytest  + '\t' + ypred)
print ("Actual Value:\n", X_test, y_test)
print ("Predicted Value:\n", y_pred_series)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


































