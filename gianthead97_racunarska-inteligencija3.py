import numpy as np

from numpy import linalg as LA



A = np.array([[2, 0], [-1, 1], [0, 2]])

b = np.array([2, 0, -2]).T
x = LA.inv(A.T @ A) @ A.T @ b

print(x)

rss = LA.norm(b - A @ x) ** 2

print(rss)
x, rss, _, _ = LA.lstsq(A, b, rcond=None)

print(x)

print(rss)
points = [[0, 1.2], [0.5, 2.05], [1, 2.9], [-0.5, 0.1]]

x = np.array([p[0] for p in points])

y = np.array([p[1] for p in points])

N = x.shape[0]
A = np.vstack((np.ones(N), x)).T

beta0, _, _, _ = LA.lstsq(A, y.T, rcond=None)
print(beta0)
points = [[3, 3, 1], [5, 3, 1], [3, 3, -1], [3, 0, 4]]

x = np.array([p[0] for p in points])

y = np.array([p[1] for p in points])

z = np.array([p[2] for p in points])

n = x.shape[0]



A =  np.vstack((np.ones(n), x, y)).T

a, b, c = LA.lstsq(A, z.T, rcond=None)[0]

(a, b, c)
points = [[2, 2.6], [-1.22, -1.7], [8.32, 2.5], [4.23, -1.6]]

x = np.array([p[0] for p in points])

y = np.array([p[1] for p in points])

n = x.shape[0]



A = np.vstack((np.ones(n), np.sin(x))).T

a, b = LA.lstsq(A, y.T, rcond=None)[0]

print('a = ', a)

print('b = ', b)
import matplotlib.pyplot as plt



t_x = np.linspace(-2, 10, 160)

t_y = a + b * np.sin(t_x)

plt.plot(t_x, t_y, color = 'r')



plt.plot(x, y, 'o', color = 'blue')
#Nemam csv fajl

import pandas as pd

# podaci = pd.read_csv('social_reach.csv')

# A = podaci

# b = 1000 * np.ones(A.shape[0])

# x = LA.lstsq(A, b, rcond=None)[0]



# x
X = np.array([[0], [0.5], [1], [-0.5]])

y = np.array([1.2, 2.05, 2.9, 0.1])
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, y)



print(lin_reg.predict(X))
print(lin_reg.intercept_, lin_reg.coef_)
df = pd.read_csv('../input/advertising.csv/Advertising.csv', index_col=0)

print(df.head())
tv = df['TV']

radio = df['radio']

newspaper = df['newspaper']

sales = df['sales']
plt.xlabel('TV')

plt.ylabel('sales')

plt.scatter(tv, sales, color='blue')
plt.xlabel('radio')

plt.ylabel('sales')

plt.scatter(radio, sales, color='red')
plt.xlabel('newspaper')

plt.ylabel('sales')

plt.scatter(newspaper, sales, color='green')
reg = LinearRegression()

x = df[['TV', 'radio', 'newspaper']]

reg.fit(x, sales)

print(reg.coef_)

print(reg.intercept_)
from sklearn import metrics



psales = reg.predict(x)

mae = metrics.mean_absolute_error(sales, psales)

mse = metrics.mean_squared_error(sales, psales)

r2 = reg.score(x, sales)



print ('MAE = ', mae)

print ('MSE = ', mse)

print ('R^2 = ', r2)
from sklearn import model_selection



x_train, x_test, y_train, y_test = model_selection.train_test_split(x, sales, test_size = 0.3)

reg = LinearRegression()

reg.fit(x_train, y_train)

r2_score_train = reg.score(x_train, y_train)

r2_score_test = reg.score(x_test, y_test)

print('R^2 train= ', r2_score_train)

print('R^2 test= ', r2_score_test)