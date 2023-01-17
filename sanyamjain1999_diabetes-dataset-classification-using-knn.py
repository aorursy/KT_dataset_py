import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
x_test = pd.read_csv("../input/Diabetes_Xtest.csv")

X_Train = pd.read_csv("../input/Diabetes_XTrain.csv")

Y_Train = pd.read_csv("../input/Diabetes_YTrain.csv")
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=17)
X = X_Train.values

Y = Y_Train.values

print(X.shape)

print(Y.shape)

Y = Y.reshape(576, )

print(Y.shape)
classifier.fit(X,Y)
xt = x_test.values

y_pred = classifier.predict(xt)
x_test = x_test.drop(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], axis = 1)
x_test['Outcome'] = y_pred
x_test.to_csv('diabetes.csv', index=True)