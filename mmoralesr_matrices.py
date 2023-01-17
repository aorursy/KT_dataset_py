# Cargamos el paquete numpy 

import numpy as np

## para generar datos de una normal estándar 

from numpy.random import randn

# una matriz de 7 por 4 

data = randn(7,4)

data
data[data<0]=0

data
arr = np.empty((8, 4))

for i in np.arange(8):

    arr[i]=i

arr
arr[[4,3,0,6]]
## selección del final hasta el principio 

arr[[-3,-5,-7]]
mt=np.arange(10).reshape(2,5)

mt

#np.array([4,3,0,6]).reshape(2,2) 
mt.T  
np.dot(mt.T,mt) 
np.dot(mt,mt.T)
# Ingresar matrices 

XtX=np.array([9,136,296,260, 

  136,2114,4176,3583,

  269,4176,8257,7104, 

  260,3583,7104,12276]).reshape(4,4)

print(XtX)

XtY=np.array([45,648,1283,1821]).reshape(4,1) 

print(XtY)
from numpy.linalg import inv

print(inv(XtX))

Bt=np.dot(inv(XtX),XtY)

Bt
arr=np.arange(10)

print(np.sqrt(arr))

np.exp(arr)
x=randn(5)

print(x)

y=randn(5)

print(y)

# compara elemento a elemento x con y toma el máximo 

np.maximum(x,y) 
x=np.arange(-5, 5, 0.01)

y=x

xs,ys=np.meshgrid(x,y)

print(xs)

ys 
print(xs.ndim)

xs.shape
z=np.sqrt(xs**2+ys**2) 
import matplotlib.pyplot as plt

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

cond = np.array([True, False, True, True, False])

## si cond es true toma x de lo contrario toma y 

np.where(cond,xarr,yarr)
arr =randn(4, 4) 

# si el valor es mayor que cero pone 2 

# de lo contrario pone -2 

np.where(arr>0,2,-2)
## si es mayor que cero pone 2 de lo contrario deja igual 

np.where(arr>0,2,arr)
### estrae la diagonal de una matrix 

XtX.diagonal() 