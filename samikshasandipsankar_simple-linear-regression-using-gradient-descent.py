import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_excel('../input/slrusinggradientdescent/companysalary.xlsx')

df
Experience = df.iloc[:, 0]

Salary = df.iloc[:, 1]

plt.scatter(Experience, Salary)

plt.show()
m = 0

c = 0

L = 0.0001  

epochs = 1000  

n = float(len(Experience)) 

for i in range(epochs): 

    Y_pred = m*Experience + c  

    D_m = (-2/n) * sum(Experience * (Salary - Y_pred))  

    D_c = (-2/n) * sum(Salary - Y_pred) 

    m = m - L * D_m  

    c = c - L * D_c  

    

print (m, c)
print (m, c)
Y_pred = m*Experience + c



plt.scatter(Experience, Salary)

plt.plot([min(Experience), max(Experience)], [min(Y_pred), max(Y_pred)], color='red') 

plt.show()