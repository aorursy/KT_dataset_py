import numpy as np
lista1 = [1,2,3]

np.array(lista1)
lista2 = [[1,2,3], [4,5,6], [7,8,9]]

np.array(lista2)
print(type(np.array(lista1)))

print(type(np.array(lista2)))
np.arange(10)
np.arange(3, 11)
np.zeros(5)
np.zeros((3,3))
np.ones(5)
np.ones((2,2))
type(np.zeros(5))
# Por default retorna 50 números.

np.linspace(8, 3)
np.linspace(1, 9, 10)
np.eye(5)
np.random.rand(4)
np.random.rand(2,3)
np.random.randn(4)
np.random.randn(2,10)
np.random.randint(23, 39, 10)
rango = np.arange(10)

rango
rango.reshape(5,2)
rango_1 = np.random.randint(2,9,50)

rango_1
rango_1.max()
rango_1.min()
# retorna el primer lugar donde se encuentra este dato, lo mismo con el argmin()

rango_1.argmax()
rango_1.argmin()
np.argmax(rango_1)
rango_1.shape
rango_1.reshape(5, 10).shape

rango_1.dtype
matriz = np.arange(10,30)

matriz
matriz[8]
matriz[3:7]
matriz[0:3] = 9 * 4
matriz
matriz_1 = matriz[3:8]

matriz_1
matriz_2 = matriz[3:8].copy()

matriz_2
matz_2d = np.array(([1,2,4],[20,45,20],[45,42,54]))

matz_2d
# Para indexar una fila, podemos hacer lo siguiente:

matz_2d[1]
# Conseguir un solo elemento en la matriz

matz_2d[1,0]
matz_2d[1][0]
matz_2d[:2]
matz_2d[:,:1]
matz_2d[1:, 1:]
matz_ceros = np.zeros((10,10))

matz_ceros
length = matz_ceros.shape

length
for i in range(10):

    matz_ceros[i] = i



matz_ceros
matz_ceros[[2,3,4]]
matz_ceros[[2,1,4,5,3]]
matriz = np.arange(1,15)

matriz
booleano = matriz > 6

booleano
matriz[booleano]
import numpy as np

arr = np.arange(0,10)
arr + arr
arr * arr
arr - arr
# Si no puede dividirse, agrega un valor nan al campo afectado

arr/arr
# Para los valores infinitos, agrega un valor inf al campo afectado

1/arr
arr**3