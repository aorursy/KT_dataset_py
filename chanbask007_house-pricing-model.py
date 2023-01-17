import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



house_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

house_data.head()
house_data.dtypes
X = house_data.drop(['price','date'],axis=1)

y = house_data['price']



X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr.fit(X_train,y_train)
y_test_a = lr.predict(X_test)

np.savetxt('output.csv',y_test_a,delimiter=',')
y_train_a = lr.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_test, y_test_a)
r2_score(y_train, y_train_a)