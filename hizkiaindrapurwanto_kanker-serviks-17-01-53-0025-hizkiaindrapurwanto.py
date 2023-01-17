import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
%matplotlib inline
data = pd.read_csv("../input/cervical-cancer/kanker serviks.csv")
data.head(5)
data.drop("intention_aggregation", axis=1 ,inplace=True)
data
cumulative = pd.get_dummies(data["intention_commitment"], drop_first=True)
cumulative
data =pd.concat([data,cumulative], axis=1)
data
data.drop("attitude_consistency",axis=1 ,inplace=True)
data
x=data.drop("attitude_spontaneity", axis=1)
y=data["attitude_spontaneity"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)


y_pred = knn.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score
predic = knn.predict(x_test)
accuracy_score(y_test, predic)
print(intention_aggregation(sc.transform([[0,30,87000]])))
