#Running a practice notebook to try out submitting kernels
import os
import numpy as np 
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# Read the data
iris = pd.read_csv('../input/Iris.csv')

train, test = train_test_split(iris, test_size = 0.3)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species

test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y =test.Species 
model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print(metrics.accuracy_score(prediction,test_y))