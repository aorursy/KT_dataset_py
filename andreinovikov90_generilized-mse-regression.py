import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/for-generalized-mse/lab1.csv", header = None, sep=" ")

df["x"]=df[0]

del df[0]

df["y"]=df[1]

del df[1]
from math import sin, exp

from matplotlib import pyplot as plt



df = df.sort_values(by="x")

plt.plot(df["x"], df["y"])

plt.show()
def approximation(x, a):

    return a[0]+a[1]*x+a[2]*x**2+a[3]**3+a[4]*x**4+a[5]*x**5+a[6]*x**6



def MSE(a,arg_nodes, val_nodes):

    if len(arg_nodes)==len(val_nodes):

        s=0

        for i in range(len(arg_nodes)):

            s+=(approximation(arg_nodes[i],a)-val_nodes[i])**2

        return s

    else:

        return -1

    

def loc_MSE(a):

    return MSE(a, df["x"], df["y"])

    



from scipy.optimize import minimize



a0 = np.array([0,0,0,0,0,0,0])

res = minimize(loc_MSE, a0, method='BFGS', options={'disp': True})

a_prime = res.x



from math import ceil



h=0.01

lower = df["x"].min()

upper = df["x"].max()

x_nodes = []

y_nodes = []

for i in range(int(ceil((upper-lower)/h))):

        x_nodes.append(lower+i*h)

        y_nodes.append(approximation(x_nodes[i], a_prime))

plt.plot(x_nodes,y_nodes, label ="Аппроксимация")

plt.plot(df["x"], df["y"], label="Входные данные")

plt.legend()

plt.grid(True)

plt.show()