# Cargamos el paquete numpy 

import numpy as np

data=np.array( [[0.9526, -0.246 , -0.8856], [0.5639, 0.2379, 0.9104]])

print(data)
# todos los elementos por 10 

print(data*10)

## suma de componente a componente  

data + data  

arr = np.array([[1., 2., 3.], [4., 5., 6.]])

print(arr)

# producto componente a componente 

print(arr*arr)

## resta componente a componente 

print(arr-arr)

## inverso de cada componente 

print(1/arr)

# raíz de cada componente 

arr**0.5 
print(data.shape," ",data.dtype)

data.ndim
ar1 = np.array([1, 2, 3], dtype=np.float64)

print(ar1,ar1.dtype)

ar2 = np.array([1, 2, 3], dtype=np.int32)

print(ar2,ar2.dtype)
ar1int=ar1.astype(np.int32)

print(ar1,ar1.dtype)

print(ar1int,ar1int.dtype)
n=5

b=np.zeros(n)

print(b)
l2=[3.1,2.6,2.0]

a=np.array(l2)

c=np.zeros_like(a) 

print(c)
u=np.ones(4) 

print(u)

u2=np.ones_like(a)

print(u2)
e=np.empty((2,3)) 

print(e) 

e2=np.empty_like(a)  
t=(2,3,4,1)

ta=np.asarray(t)

print(ta,type(ta))

t2=((2,3),(4,1)) 

ta2=np.asarray(t2)

print(ta2,type(ta2))
np.arange(2,10,0.5)
id=np.eye(4)

print(id)

id2=np.identity(3)

print(id2)
a=np.linspace(0,5,15) 

print(len(a))

print(a,type(a)) 
a[1:8:2]
a[1:-1]
b=a[1:-1]

print(b[0])

b[len(b)-1]
a[::4]
# datos con distribución normal ertándar 

# en ua matriz de 5 filas y 4 columnas   

np.random.seed(2341) 

arr = np.random.randn(5, 4)

print(np.round(arr,2))

# suma

print(arr.sum())

print(np.sum(arr)) 

# media 

print(arr.mean())

print(np.mean(arr))

# maximo 

arr.max()

print("Maximo", np.max(arr))

# mínimo

arr.min()
## suma por columna   

#print(arr.sum(axis=0))

#print("Suma de las columnas ", arr.sum(0))

## suma por fila  

#print("Suma de las filas ", arr.sum(1))

#print(arr.sum(axis=1))

## maxino por columna  

print(arr.max(axis=0))

## suma por fila 

#print(arr.max(1))
ar1=np.array([2,3,1,5,8])

# Suma acumulada 

print(ar1.cumsum())

# producto acumulado 

print(ar1.cumprod())

print(np.shape(arr))

print(arr.size) 

len(arr)
np.median(ar1)
#print(ar1.std())

#np.std(ar1)

print(ar1.var())

#print(np.sqrt(ar1.var()))
np.sum( (ar1-ar1.mean())**2 )/len(ar1) 
print(ar1.var(ddof=1)) 
np.sum( (ar1-ar1.mean())**2 )/(len(ar1)-1)
np.random.seed(2341) 

arr2 = np.round( np.random.randn(6, 5), 2) 

# cuartiles 

q=np.arange(0,1.25,0.25)  

print(q)

np.quantile(arr2,q)
q=np.arange(0,1.1,0.1)

print(q)

np.quantile(arr2,q)
np.quantile(arr2,np.array([0.12,0.3,0.65,0.8]) )
t11=np.array( [105,221,183,186,121,181,180,143,

97,154,153,174,120,168,167,141,

245,228,174,199,181,158,176,110,

163,131,154,115,160,208,158,133,

207,180,190,193,194,133,156,123,

134,178,76,167,184,135,229,146,

218,157,101,171,165,172,158,169,

199,151,142,163,145,171,148,158,

160,175,149,87,160,237,150,135,

196,201,200,176,150,170,118,149] ) 

np.quantile(t11,np.arange(0,1.25,0.25)) 
n=len(t11)

print(n)

p=25

## Lp

Lp=((n+1)*p)/100

print(Lp)

## Parte entera de Lp

PeLp=((n+1)*p)//100

## Parte decimal de Lp

PdLp=Lp-PeLp

print(PdLp)

t11o=np.sort(t11)

print(t11o)

print(t11o[PeLp-1] + PdLp*(t11o[PeLp]-t11o[PeLp-1]) )

#print(t11o[PeLp-1])

#print(143+0.25*(145-143))

#np.quantile(t11,np.arange(0,1.25,0.25), interpolation='linear')

#help(np.quantile)

#Example(np.quantile)

#?np.quantile
np.quantile(t11,np.arange(0,1.25,0.25), interpolation='midpoint')