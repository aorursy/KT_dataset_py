# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# Ecuación del eje X

FX = lambda x, k1, k2: np.sin(k2*np.exp(k1*x))

# Ecuación del eje Y

FY = lambda y, k3: np.exp((-1/k3)*y)
# Hallar k1 y k2, no me llega a funcionar esto ;-;

freq_i = np.pi/512

k2 = 512*freq_i/np.log(np.pi/freq_i)

k1 = freq_i/k2

print("K2: {} y k1: {}".format(k2, k1))
x=np.linspace(0,1.5*np.pi,512)

y=np.linspace(0,20,512)

X,Y=np.meshgrid(x,y)

Z=FX(X,1.13309,k2)*FY(Y,10/3)



plt.pcolormesh(X,Y,Z, cmap='gray')

plt.colorbar()
cg = lambda x, k1, k2, D: k1*k2*np.exp(k1*x/930)/(2*np.pi*np.arctan(0.025/D))

cp = lambda x, k1, k2: k1*k2*np.exp(k1*x)



k1 = 0.01114

k2 = 0.94002



D = np.array([100,200,300])



ricardo = np.array([686, 629, 547])

diego = np.array([688, 563, 471])

xareni = np.array([540, 515, 439])

team = [ricardo, diego, xareni]



list_cg = [cg(x,k1,k2,D) for x in team]

list_cp = [cp(x,k1,k2) for x in team]



print("Ciclo por pixel de un metro a 3")

print("Ricardo: {} \nDiego: {} \nXareni: {}"

     .format(list_cp[0], list_cp[1], list_cp[2]))



print("Ciclo por grado de un metro a 3")

print("Ricardo: {} \nDiego: {} \nXareni: {}"

     .format(list_cg[0], list_cg[1], list_cg[2]))



#930x605
# f(n)

x = np.array([0,1,2,3])

f = np.array([3,3,3,3])



plt.stem(x, f)

plt.title("f(n)")

plt.show()



# h(n)

plt.title("h(n)")

h = lambda x: 3-x

    

plt.stem(x, h(x))

plt.show()



# g(n)

plt.title("g(0)=3")

plt.stem([0], [3])

plt.show()



plt.title("g(1)=5")

plt.stem([0,1], [2,3])

plt.show()



plt.title("g(2)=6")

plt.stem([0,1,2], [1,2,3])

plt.show()



plt.title("g(3)=6")

plt.stem([0,1,2,3], [0,1,2,3])

plt.show()



plt.title("g(4)=3")

plt.stem([1,2,3], [0,1,2])

plt.show()



plt.title("g(5)=1")

plt.stem([2,3], [0,1])

plt.show()



plt.title("g(6)=0")

plt.stem([3], [0])

plt.show()
