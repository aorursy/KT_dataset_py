import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
data = pd.read_csv("../input/insurance_data.csv")
data.shape
data.head()
X = data[['age']]

y = data.bought_insurance
plt.scatter(X,y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
model.score(X_test,y_test)
confusion_matrix(y_test,y_pred)
