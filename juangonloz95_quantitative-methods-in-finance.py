## Script for options



# Needed libraries

from numpy import random as rd

from math import *

import matplotlib.pyplot as plt
# defining graph function

def graph_f(f,X_inf,X_sup,N):

    x_axis = []

    y_axis = []

    for i in range(N):

        deltax = float(X_sup-X_inf)/N

        x = X_inf+i*deltax

        x_axis.append(x)

        y = f(x)

        y_axis.append(y)

    plt.plot(x_axis,y_axis)

    plt.grid(True)
# defining option funtions

def long_call(s):

    return max(s-K,0)

def short_call(s):

    return -max(s-K,0)

def long_put(s):

    return max(K-s,0)-p

def short_put(s):

    return -max(K-s,0)

def bull_spread(s):

    return max(K1-s,0)-max(K3-s,0)

def bear_spread(s):

    return max(s-K3,0)-max(s-K1,0)

def straddle(s):

    return max(s-K,0)+max(K-s,0)

def strangle(s):

    return max(s-Kc,0)+max(Kp-s,0)

def butterfly(s):

    return max(s-K1,0)-2*max(s-K2,0)+max(s-K3,0)
X_inf,X_sup,N = 0,150,150 # Parámetros para la gráfica



graph_f(long_put,X_inf,X_sup,N)
## Importar libreria de matemáticas



from math import *
## Estableciendo parámetros y variables



S0, K, r, sigma, T = 10, 11.5, 0.1, 0.21, 1

N = 4
K
## Hallando los valores de u,d,p



dt = float(T)/N

u = exp(sigma*sqrt(dt))

d = 1/u

p = (exp(r*dt)-d)/(u-d)

print(u,d,p)
## creando la matriz del arbol



tree = [[0.0 for i in range(N+1)] for j in range(N+1)]

tree ## comprobando la matriz
## asignando valores al arbol

i = 0 # row i - column j



for i in range(N+1):

    if(i==0):

        tree[i][i] = S0

    else:

        tree[i][i] = tree[i-1][i-1]*d

        

    for j in range(i+1,N+1):

        tree[i][j] = tree[i][j-1]*u



print("el valor de Suu es:",tree[1][N], "mientras que el Sdd es:", tree[N][N])
## calculando el valor de la opción



Vtree = [[0.0 for i in range(N+1)] for j in range(N+1)]

j = N

for i in range(N,-1,-1):

    Vtree[i][j] = max(K-tree[i][j],0)

for j in range(N-1,-1,-1):

    for i in range(j,-1,-1):

        Vtree[i][j] = exp(-r*dt)*((p*Vtree[i][j+1]) + ((1-p)*Vtree[i+1][j+1]))

print("El valor de la put es:", Vtree[0][0])