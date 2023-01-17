#Import packages and load the dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets

from sklearn.datasets import load_iris
iris = load_iris()

data =  np.c_[iris['data'], iris['target']]

df = pd.DataFrame(data, columns=iris["feature_names"]+["target"])
df.head()
from sklearn.model_selection import train_test_split

features = df[iris["feature_names"]]

labels = df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size = 0.30)
# write your Code Here 

from sklearn.ensemble import RandomForestClassifier

# n_estimators = 100

rf=RandomForestClassifier(n_estimators=100)

#random_state any integer number

rf = rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
from sklearn import metrics



print("Accuracy of the model is:",metrics.accuracy_score(Y_test, y_pred))
print("thank you guys")