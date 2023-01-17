import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/Iris.csv')

x= dataset.iloc[:, 1:5].values

y= dataset.iloc[:, 5].values



from sklearn.preprocessing import StandardScaler

from sklearn.compose import make_column_transformer 

scaler = StandardScaler()

x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
#Check you results by looking at the confusion matrix or by indivdually analyzing the test and predicted values

print(cm)

print(y_test)

print(y_pred)