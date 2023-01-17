import numpy as np

import pandas as pd

import statsmodels.api as sm



df = pd.read_csv('../input/students-data-for-mlr/amsPredictionSheet1-201009-150447.csv')

df
y = df['ESE']

x = sm.add_constant(df[['Attendance', 'MSE', 'HRS']])

z = df[['Attendance', 'MSE', 'HRS']]

print(x, y, z)
x_transpose = x.T

x_transpose
a = x_transpose.dot(x)

a_inverse = np.linalg.inv(a)

a_inverse
B = a_inverse.dot(x_transpose.dot(y))

B
def predict_y(x1, x2, x3):

    y_hat = B[1]*x1 + B[2]*x2 + B[3]*x3 + B[0]

    return y_hat;
ESE_pre = predict_y(df['Attendance'], df['MSE'], df['HRS'])

ESE_pre
r_square = 1 - sum((y-ESE_pre)*(y-ESE_pre))/sum((y-y.mean())*(y-y.mean()))

r_square
import matplotlib.pyplot as plt

plt.scatter(df['HRS'], y)

plt.scatter(df['HRS'], ESE_pre)

plt.show()
plt.scatter(df['MSE'], y)

plt.scatter(df['MSE'], ESE_pre)

plt.show()
plt.scatter(df['Attendance'], y)

plt.scatter(df['Attendance'], ESE_pre)

plt.show()