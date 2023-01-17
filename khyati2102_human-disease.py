import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the data

heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart.shape

#heart.head()

#heart.info()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(heart,heart['target'],stratify=heart['target'],random_state=0)
from sklearn.preprocessing import StandardScaler

scaled=StandardScaler()

X_train=scaled.fit_transform(X_train)

X_test=scaled.fit_transform(X_test)
from sklearn.svm import LinearSVC

lsvc=LinearSVC()

lsvc.fit(X_train,y_train)
y_pred=lsvc.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy score : {:.2f}".format(accuracy_score(y_test,y_pred)))