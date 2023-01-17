import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
dataset.head()
X=dataset.iloc[:, 2:-1].values
y=dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cm
accuracy_score(y_test, y_pred)