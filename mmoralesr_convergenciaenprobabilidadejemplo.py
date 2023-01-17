from scipy import stats

# Define una variable aleatoria Uniforme 

u=stats.uniform() 

# función que genera una muestra de tamaño i y toma el máximo 

xn = lambda i: u.rvs(i).max()

# máximo de una muestra de tamaño 5 

#print(u.rvs(5))

xn(5)
import numpy as np

np.mean([xn(60) > 0.95 for j in range(1000)]) 
np.log(0.01)/np.log(0.95)
np.mean([xn(90) > 0.95 for i in range(1000)])
maximo=np.array(range(1,100,1),dtype='float64')

#print(maximo)

for x in range(1,100,1):

    maximo[x-1]=np.mean( [xn(x)>0.95 for i in range(1000)] )

x=range(1,100,1)
import matplotlib.pyplot as plt

plt.xlabel('$n$')

plt.ylabel('$X_n$')

plt.plot(x,maximo) 

plt.hlines(0.99,xmin=0,xmax=100,linestyles='dashed')

plt.show()