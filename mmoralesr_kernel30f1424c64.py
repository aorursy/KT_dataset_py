# cargamos el paquete numpy como np

import numpy as np

# genera un arreglo de 4 ceros  

ceros=np.zeros(4)

print(ceros,type(ceros))

# una matriz de dos filas tres columnas de ceros 

np.zeros((2,3))
np.ones(4)
np.ones((2,3))
np.arange(3)
a=np.arange(-2.5,4,0.5)

print(a)
b=np.linspace(0, 15, 10)

b
np.random.rand(10)
np.repeat(3, 4)
np.repeat([1,2,3], [3,4,5])
x = np.array([[1,2],[3,4]])

print(x)

np.repeat(x, 2, axis=0)

#np.repeat(x, 2, axis=1)
np.repeat(np.arange(1, 5), 4)
a = np.array([0, 1, 2])

np.tile(a, 3)