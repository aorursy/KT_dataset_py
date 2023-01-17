import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_excel('../input/demo-marks/marks.xlsx')

df
x = df['MSE']

y = df['ESE']

plt.scatter(x, y)

plt.show()
m = 0

c = 0

L = 0.01

count = 20000

n = float(len(x))



for i in range(count):

    ybar = m*x + c

    m = m - (L/n)*sum(x*(ybar-y))

    c = c - (L/n)*sum(ybar-y)

print(m, c)
plt.scatter(x,y)

plt.plot(x, ybar, color = 'red')

plt.show()
ymean = np.mean(y)

r_square = 1- sum((y-ybar)*(y-ybar))/sum((y-ymean)*(y-ymean))

r_square