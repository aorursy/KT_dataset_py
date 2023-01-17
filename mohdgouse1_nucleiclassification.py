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
df = pd.read_csv('../input/data.csv')
df.head()
list(set(df.Label))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split #to split the dataset for training and testing
train , test = train_test_split(df, test_size = 0.3)
X_train = train.drop(["Label"] , axis = 1)

Y_train = train.Label



X_test = test.drop(["Label"] , axis = 1)

Y_test = test.Label
Logistic_regressor = LogisticRegression()

Logistic_regressor.fit(X_train, Y_train)
Logistic_prediction = Logistic_regressor.predict(X_test)
from sklearn.metrics import confusion_matrix 

from sklearn import metrics #for checking the model accuracy
Logistic_cm = confusion_matrix(Y_test, Logistic_prediction)

Logistic_accuracy = metrics.accuracy_score(Y_test, Logistic_prediction)
print("The accuracy of the classifier is: {}".format(Logistic_accuracy))
print("The Confusion matrix of the classifier is: {}".format(Logistic_cm))