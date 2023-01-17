import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

%matplotlib inline

iris = load_iris()
print(iris.DESCR)
features = pd.DataFrame(iris.data, columns = iris.feature_names)
target = pd.DataFrame(iris.target, columns=['Target'])
data = pd.concat([features, target], axis=1)
data
data2 = data.corr('pearson')
data2
abs(data2.loc['Target']).sort_values(ascending=False)
x = data['petal width (cm)']
y = data['Target']
x = np.array(x/x.mean())
y = np.array(y/y.mean())
n = int(0.8*len(x))
x_train = x[:n]
y_train = y[:n]

x_test = x[n:]
y_test = y[n:]
plt.plot(x_train, y_train, 'b.')
plt.plot(x_test, y_test, 'g.')
def hypothesis(a,b,x):
  return a*x + b
def error(a,b,x,y):
  e = 0
  m = len(y)
  for i in range(m):
    e +=np.power((hypothesis(a,b,x[i])-y[i]),2)
  return (1/(2*m))*e  
def step_gradient(a,b,x,y,learning_rate):
  grad_a = 0
  grad_b = 0
  m = len(x)
  for i in range(m):
    grad_a += 1/m * (hypothesis(a,b,x[i])-y[i]) * x[i]
    grad_b += 1/m * (hypothesis(a,b,x[i])-y[i])

  a = a- (grad_a * learning_rate)
  b = b- (grad_b * learning_rate)
  return a,b  

def descend(initial_a, initial_b, x,y, learning_rate,iterations):
  a = initial_a
  b = initial_b
  for i in range(iterations):
    e = error(a,b,x,y)
    if i%1000==0:
      print(f"Error: {e}, a: {a}, b: {b}")
    a,b = step_gradient(a,b,x,y,learning_rate)
  return a,b    

a = 0
b = 1
learning_rate = 0.01
iterations = 10000
final_a, final_b = descend(a,b,x_train, y_train, learning_rate, iterations)

print(error(a,b,x_train, y_train))
print(error(final_a, final_b, x_train, y_train))
print(error(final_a, final_b, x_test, y_test))
plt.plot(x_test, y_test, 'r.', x_test, hypothesis(final_a, final_b, x_test), 'g')

