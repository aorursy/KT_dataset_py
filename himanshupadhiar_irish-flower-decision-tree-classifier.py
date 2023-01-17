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
import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
dataset = "../input/Iris.csv"

df_data = pd.read_csv(dataset)

df_data.head()
# Check the number of each Species.

df_data["Species"].value_counts()
# Store the independent varaibles as an array.

X = df_data[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

print(X[0:5])
# Store the dependent varaibles as an array.

y = df_data["Species"].values

print(y[0:5])
# Create a train test split.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)

print('Train Set : ', X_train.shape, y_train.shape)

print('Test Set : ', X_test.shape, y_test.shape)
# Create and train the Decision Tree Classifier model.

model = DecisionTreeClassifier(criterion="entropy")

model.fit(X_train, y_train)
# Predict the species of the test data set.

y_hat = model.predict(X_test)



# Check the actual and predicted values

print(y_test[0:5])

print(y_hat[0:5])
from sklearn import metrics

acc = metrics.accuracy_score(y_test, y_hat)

print('Model Accuracy : ', acc)