import numpy as np
import pandas as pd


arquivo = pd.read_csv("../input/ex1_train_data.csv", sep=";")
arquivo

a = 1

for i in range(0, len(arquivo.index)):
    b = 0
    
    x = arquivo['x'][i]
    y = arquivo['y'][i]
    
    result = a * x + b
#     while result < y:
#         b = b + 1
#         result = a * x + b
        
    print('b', b)
    print('result', result)
    print('y', y)
    print('\n\n\n')
    
import pandas as pd
import numpy as np
import os
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

R = 10

X = pd.DataFrame(arquivo[["x"]])
y = pd.DataFrame(arquivo[["y"]])

linearRegressor = LinearRegression()

model = []

for i in range(1,R):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
    
    #Regressão Linear
    model.append(linearRegressor.fit(X_train, y_train))
    y_pred = linearRegressor.predict(X_test)
    
    print('y_pred', y_pred)
X = pd.DataFrame(arquivo[["x"]])
y = pd.DataFrame(arquivo[["y"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)


#Intercepto ou Coeficiente Linear
# print(linearRegressor.intercept_)
#Coeficiente Angular (slope)
# print(linearRegressor.coef_)

#Previsão
y_pred = linearRegressor.predict(X_test)
print(y_pred)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

from matplotlib import pyplot as plt

plt.scatter(x=arquivo['x'], y=arquivo['y'])
plt.show
from matplotlib import pyplot as plt

plt.scatter(x=arquivo['x'], y=y_pred)
plt.show
import pandas as pd

train_data = pd.read_csv('../input/ex1_train_data.csv', delimiter=';')

def mse(Y, Y_pred):    
    result = 1/len(Y) * sum((Y - Y_pred) ** 2)
    return result

best_a = None
best_b = None
best_mse = None

a = 0
b = 0

L = 0.00001

X = train_data['x']
Y = train_data['y']

n = len(train_data['y'])

epochs = 10000 #epocas

for i in range(1, epochs):
    Y_pred = train_data['x'] * a + b

#     new_mse = mse(train_data['y'], Y_pred)

#     if best_mse is None or new_mse < best_mse:            
#         best_mse = new_mse
#         best_a = a
#         best_b = b

    a = a - L * ((-2/n) * sum(X * (Y - Y_pred))) #derivado do erro (proprio mse)
    b = b - L * ((-2/n) * sum(Y - Y_pred))
        

# print('best_mse = %.3f, best_a = %f, best_b = %f' % (best_mse, best_a, best_b))

Y_pred = train_data['x'] * a + b
new_mse = mse(Y, Y_pred)
print('mse', new_mse)
print('a', a)
print('b', b)

