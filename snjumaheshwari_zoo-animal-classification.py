import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



import os

print(os.listdir('../input'))
df = pd.read_csv('../input/zoo.csv')



df.head()
features = list(df.columns)

print(features)
features.remove('class_type')

features.remove('animal_name')



print(features)
X = df[features].values.astype(np.float32)

Y = df.class_type





print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, Y_train)

print("training accuracy :", model.score(X_train, Y_train))

print("testing accuracy :", model.score(X_test, Y_test))
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

print("training accuracy :", model.score(X_train, Y_train))

print("testing accuracy :", model.score(X_test, Y_test))



from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier()

model.fit(X_train, Y_train)

print("training accuracy :", model.score(X_train, Y_train))

print("testing accuracy :", model.score(X_test, Y_test))

from sklearn.svm import SVC



model = SVC(kernel = "rbf", C = 1.0, gamma = 0.1)

model.fit(X_train, Y_train)

print("training accuracy :", model.score(X_train, Y_train))

print("testing accuracy :", model.score(X_test, Y_test))
from sklearn.ensemble import AdaBoostClassifier



model = AdaBoostClassifier()

model.fit(X_train, Y_train)

print("training accuracy :", model.score(X_train, Y_train))

print("testing accuracy :", model.score(X_test, Y_test))