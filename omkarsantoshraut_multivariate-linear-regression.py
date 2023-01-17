import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model



df = pd.read_csv('../input/markswithattendence/markswithattendence.csv')

df
import math

attendance = df.Attendance.median()

mse_marks = df.MSE.median()

ese_marks = df.ESE.median()

print(attendance, mse_marks, ese_marks)
df.Attendance = df.Attendance.fillna(attendance)

df.MSE = df.MSE.fillna(mse_marks)

df.ESE = df.ESE.fillna(ese_marks)
x = df[['Attendance', 'MSE']]

y = df['ESE']



reg_obj = linear_model.LinearRegression()

reg_obj.fit(x, y)
print(reg_obj.coef_)
print(reg_obj.intercept_)
print(reg_obj.predict([[90, 19]]))
y_predict = reg_obj.predict(x)

y_predict
x1 = df['Attendance']

x2 = df['MSE']



plt.scatter(x1, x2, y)

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection = '3d')

ax.scatter(x1, x2, y, color = 'red', marker='o')

ax.scatter(x1, x2, y_predict, color = 'blue', marker = '^')

ax.set_xlabel('Attendance')

ax.set_ylabel('MSE')

ax.set_zlabel('ESE')

plt.show()

r_square = 1-(sum((y-y_predict)*(y-y_predict))/ sum((y - y.mean())*(y - y.mean())))

r_square