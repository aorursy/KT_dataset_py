import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def fibbonaciSequence(n):
    if n == 0: return 1
    if n == 1: return 2
    return fibbonaciSequence(n - 1) + fibbonaciSequence(n - 2)

def data():
    list = []
    for x in range(25):
        list.append({ 'number': x, 'point': fibbonaciSequence(x) })
    return list

df = pd.DataFrame.from_dict(data())

data = df.groupby('number').point.mean().reset_index()
y = data['point']
x = data['number']
X = x.values.reshape(-1, 1)
# Create linear regression model
def regression():
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    return regr
# Prediction Data
linear_model = regression()
y_predict = list(map(lambda x: x * linear_model.coef_ + linear_model.intercept_, X))
print(fibbonaciSequence(len(y_predict)), len(y_predict), y_predict[len(y_predict) -1][0])
from sklearn.linear_model import LogisticRegression

# Create linear regression model
def regression():
    regr = LogisticRegression()
    regr.fit(X, y)
    return regr
# Prediction Data
logistic_model = regression()
import matplotlib.pyplot as plt

# graph plotting commands
plt.scatter(X, y)
plt.plot(X, y_predict, 'm')
# graph plotting commands
plt.plot(logistic_model.predict(X), 'black')
plt.show()
print(
    "Linear:", int(linear_model.score(X, y) * 100), "% ", 
    " Logistic:", int(logistic_model.score(X, y) * 100), "%"
)