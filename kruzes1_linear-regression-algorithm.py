import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression
df = pd.read_csv("../input/headbrainlr/headbrain.csv")

df.shape
ML_model = LinearRegression()
x = df.iloc[:,0:1].values

y = df.iloc[:,1].values

ML_model.fit(x,y)
m = ML_model.coef_

c = ML_model.intercept_
y_predict = m*x+c
y_predict
y
import matplotlib.pyplot as plt

plt.scatter(x,y)

plt.plot(x,y_predict, c="red")

plt.show()

y_predict = ML_model.predict(x)
h = 3000

w = ML_model.predict([[h]])
plt.scatter(x,y)

plt.plot(x,y_predict, c="red")

plt.scatter([h],w,c="Black")

plt.show()