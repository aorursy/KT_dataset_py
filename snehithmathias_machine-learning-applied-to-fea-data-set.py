import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv(r"../input/fem-simulations/1000randoms.csv")
data.head()
X = ['ecc', 'N', 'gammaG', 'Esoil', 'Econc', 'Dbot', 'H1', 'H2', 'H3']
Y = ['Mr_t', 'Mt_t', 'Mr_c', 'Mt_c']
X = data[X]
Y = data[Y]
X
Y
# Linear Regression
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=20)
Linear = LinearRegression()
Linear.fit(X_train, Y_train)
print(Linear.score(X_train, Y_train))
y_test = Linear.predict(X_test)
#print(y_test)
#print(Linear.score(X_test, y_test))
y_test = pd.DataFrame(y_test)
Y_test = pd.DataFrame(Y_test)
Y_test
Y_test_0 = Y_test.drop(['Mt_t','Mr_c','Mt_c'], axis=1)
#y_test
y_test_0 = y_test.drop([1,2,3], axis=1)
#y_test_0 
plt.plot(Y_test_0, y_test_0)
plt.show()
# Random Forest Regresion
RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
print(RF.score(X_train, Y_train))
y_t = RF.predict(X_test)
print(RF.score(X_test, y_t))
y_t = pd.DataFrame(y_t)
y_t
y_t_1 = y_t.drop([1,2,3], axis=1)
Y_t_1 = Y_test.drop(['Mt_t','Mr_c','Mt_c'], axis=1)
plt.plot(y_t_1, Y_t_1)
plt.show()
