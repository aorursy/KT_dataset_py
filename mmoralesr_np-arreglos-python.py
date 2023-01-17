# Cargamos el paquete numpy 

import numpy as np

l2=[7,8,9] # se crea la lista l2 

#l2=[7,8,9,"l"] # se crea la lista l2 

# convertimos la lista l3 en un arreglo 

a=np.array(l2)

print(a,type(a))
n=5

b=np.zeros(n)

print(b)
c=np.zeros_like(a)

print(c)
a=np.linspace(0,5,15)

print(len(a))

print(a,type(a)) 
a[1:8:2]
a[1:-1]
b=a[1:-1]

print(b[0])

b[len(b)-1]
a[::4]
# para tener disponibles las funciones coseno y exponencial 

from math import cos, exp

n=5

x=np.linspace(0,1,n)

print(x,type(x))

# definimos la función f

def f(x):

    return(x**2*cos(x)*exp(x)+2)

# creamos un arreglo y 

y=np.array( [f(i) for i in x ] )  # note el uso de for para crear la lista  

print(y,type(y))
def f2(x):

    return(x**2*np.cos(x)*np.exp(x)+2)

y=f2(x)



print(y)
# Esto no trabaja porque f usa las funciones de math 

# que solo trabajan sobre elementos unidimensionales

#f(x) 
import numpy as np

# se define la función 

def f3(t):

    return t**2*np.exp(-t**2)

# se generan los valores de $t$

t=np.linspace(0,3,51)

# calcular los valores de f(t)

y=f3(t)

# importar el paquete para graficar 

from matplotlib.pyplot import *

plot(t,y)

show()
plot(t, y,label=r'$t^2*exp(-t^2)$')

xlabel('t')

ylabel('y')

legend(loc='best')

axis([0, 3, -0.05, 0.4]) # [tmin, tmax, ymin, ymax]

title('Mi primer gráfico con Matplotlib')

show()