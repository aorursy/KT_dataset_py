import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/heights-and-weights/data.csv")
data.head()
data.dropna()
data.mean()
from sklearn.model_selection import train_test_split

X = data.iloc[:,:-1].values
y = data.iloc[:,1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
predict = regression.predict(X_test)
data_output = pd.DataFrame({"actual":y_test,"predicted":predict})
data_output
plt.scatter(X_train,y_train,color="red")

plt.plot(X_train,regression.predict(X_train),color="blue")

plt.title("height vs weight")

plt.xlabel("h")

plt.ylabel("w")

plt.show()
plt.scatter(X_test,y_test,color="red")

plt.plot(X_train,regression.predict(X_train),color="blue")

plt.title("height vs weight")

plt.xlabel("h")

plt.ylabel("w")

plt.show()