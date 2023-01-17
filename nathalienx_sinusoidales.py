import numpy as np

import pylab as plt
def sinus (x):

    # compléter la fonction pour retourner la valeur sin(x)

    # solution :

    return np.sin(x)



## TEST

print("sin de 0 :", sinus(0))

print("sin de pi/2 :", sinus(np.pi/2))

print("sin de pi :", sinus(np.pi))
plt.plot(0, sinus(0), ".")
X1 = [i for i in np.arange(0, 4*np.pi, 1)]       # liste des nombres entre 0 inclus et 4pi exclus, avec un pas de 1  

X2 = [i for i in np.linspace(0, 4*np.pi, 10)]     # liste des nombres entre 0 inclus et 4pi inclus, en répartissant sur 10 points



print('X1 : ', X1)

print('X2 : ', X2)
Y1 = sinus(X1)

Y2 = sinus(X2)
plt.plot(X1, Y1, 'x', color='red')

plt.plot(X2, Y2, '+', color='blue')

plt.show()
X = [i for i in np.linspace(0, 4*np.pi, 100)]     # liste des nombres entre 0 inclus et 4pi inclus, en répartissant sur 100 points

Y = sinus(X)



plt.plot(X, Y)
A = 3

def sinus_A (x):

    return A*np.sin(x)



Y_sinus = [sinus(x) for x in X]

Y_sinus_A = [sinus_A(x) for x in X]

plt.plot(X, Y_sinus, 'red')

plt.plot(X, Y_sinus_A, 'blue', ls=':')
T = 4

def sinus_T (x):

    return np.sin( (2*np.pi) / T * x)



Y_sinus_T = [sinus_T(x) for x in X]

plt.plot(X, Y_sinus, 'red')

plt.plot(X, Y_sinus_T, 'blue', ls=':')
B = 1

def sinus_B (x):

    return np.sin( x - B )



Y_sinus_B = [sinus_B(x) for x in X]

plt.plot(X, Y_sinus, 'red')

plt.plot(X, Y_sinus_B, 'blue', ls=':')
A = 1.1

T = 6

B = 0.1

def sinus_2 (x):

    return A*np.sin( (2*np.pi) / T * x - B)



Y_sinus_2 = [sinus_2(x) for x in X]

plt.plot(X, Y_sinus, 'red')

plt.plot(X, Y_sinus_2, 'blue', ls=':')