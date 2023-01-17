# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()
df = data.values

X = df[:,1:5]

Y = df[:,5]



#Splitting the data

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=.20, random_state=7)
clf = GaussianNB()  #Naive Bayes



#Fit the model

clf.fit(X_train, Y_train)



#Predicting

pred_clf = clf.predict(X_validation)
print("Gaussian Naive Bayes model accuracy(in %):", accuracy_score(Y_validation, pred_clf)*100)