import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_excel('../input/studentmarks/studentmarks.xlsx')
df
x = df['MSE']

x
y = df['ESE']

y
plt.scatter(x,y)

plt.show()
x_mean = np.mean(x)

x_mean
y_mean =np.mean(y)

y_mean
m = sum((x-x_mean)*(y-y_mean))/sum((x-x_mean)*(x-x_mean))

m
c = y_mean - m*x_mean

c
def predict_marks(x):

    y_predicted = 1.6321516393442619*x+30.6663524590164

    return y_predicted



y_predicted = predict_marks(x)

y_predicted
plt.scatter(x,y)

plt.plot(x, y_predicted, color = 'red')

plt.show()
demo_mark = 10

demo = predict_marks(demo_mark)

demo
demo1 = predict_marks(20)

demo1
r_square = 1- sum((y-y_predicted)*(y-y_predicted))/sum((y-y_mean)*(y-y_mean))

r_square