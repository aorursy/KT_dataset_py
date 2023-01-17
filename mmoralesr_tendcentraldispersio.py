

import numpy as np

np.random.seed(2341) 

# vector de 25 datos con distribución normal ertándar 

arr = np.random.randn(25)

print("Datos ", np.round(arr,2))

# suma

print("Suma ", arr.sum())

print(np.sum(arr)) 

# media 

print("Media ",arr.mean())

print(np.mean(arr))

# maximo 

arr.max()

print("Maximo", np.max(arr))

# mínimo

arr.min()
# Datos como una matriz 

#arr = np.random.randn(5,5)

## suma por columna   

#print(arr.sum(axis=0))

#print("Suma de las columnas ", arr.sum(0))

## suma por fila  

#print("Suma de las filas ", arr.sum(1))

#print(arr.sum(axis=1))

## maxino por columna  

#print(arr.max(axis=0))

## suma por fila 

#print(arr.max(1))
## ejemplo con poquitos datos 

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