import pandas as pd

import numpy as np
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()
data.describe()
data.tail()
y = data.target.values

y
x_data = data.drop(['target'], axis = 1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 200)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train, y_train)
y_predicted = lr.predict(x_test)
print("Training Score:",lr.score(x_train,y_train))

print("Test Score:",lr.score(x_test,y_test))