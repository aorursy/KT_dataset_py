import numpy as np





"""

TODO: Crear matriz A

"""

A = np.array([

              [1, 2, 3],

              [4, 5, 6],

              [7, 8, 9]

])



print(type(A))

print(A.dtype)

print(A.shape)
"""

TODO: Crear matriz de ceros

"""



B = np.zeros((3, 3))



assert(B.shape == (3, 3))

B
"""

TODO: Crear matriz identidad

"""

B = np.eye(3)



B
"""

TODO: Completar 

"""



for i in range(np.shape(A)[0]):

    for j in range(np.shape(A)[1]):

        print(A[i][j])

A = np.array([

    [1, 2, 3],

    [4, 5, 6],

    [7, 8, 9]

])



B = np.array([

    [1, 1, 1],

    [0, 1, 1],

    [0, 0, 1]

])
"""

TODO: Completar código

"""



#Método 1: iterativamente



C = np.zeros((3, 3))



for i in range(A.shape[0]):

  for j in range(A.shape[1]):

    C[i][j] = A[i][j] + B[i][j]



print(C)

# Método 2: vectorialmente (mucho mejor!)

C = A + B



print("\n")

print(C)
"""

TODO: Calcular el producto de A y B

"""



# Método 1



C = np.zeros((3, 3))

for i in range(A.shape[0]):

    for j in range(A.shape[1]):

        for k in range(A.shape[1]):

            C[i][j] += A[i][k]*B[k][j]





# Método 2: mejor

C = A @ B





assert(np.allclose(C, np.array([

    [1, 3, 6],

    [4, 9, 15],

    [7, 15, 24]

])))
A = np.array([

    [.1, 0, 0],

    [0, .1, 0],

    [0, 0, .1],

], dtype="float")



B = A + A + A



C = np.array([

    [.3, 0, 0],

    [0, .3, 0],

    [0, 0, .3],

], dtype="float")
print(f"B = \n {B}")

print(f"C = \n{C}")

result = B == C



type(result)
(B == C).all()
# Esto es lo mismo que hacer 

# np.isclose(B, C).all()

np.allclose(B, C)
A = None



A
A = None



A
"""

TODO: Completar código

""" 

A = np.array([

    [1, 2, 3],

    [4, 5, 6],

    [7, 8, 9]

])



fila = A[1]

assert(np.allclose(fila, np.array([4, 5, 6])))
columna = A[:, 2]



assert(np.allclose(columna, np.array([3, 6, 9])))
submatriz = submatriz = A[:2, :2]



assert(np.allclose(

    submatriz, 

    np.array([

       [1, 2],

       [4, 5]

    ])

))
# range es un iterador ... para convertirlo en una lista concreta, hay que hacer esto



nums = list(range(1, 10))



L = np.array(nums)



"""

TODO: Completar código

"""

A = np.reshape(L, (3, 3))





assert(np.allclose(

    A, np.array([

    [1, 2, 3],

    [4, 5, 6],

    [7, 8, 9]

])))
v = A[:, 0]



print(v @ v.T)



v = np.reshape(v, (3, 1))



print(v.shape)

print(v.shape == (1,3))

print(v.shape == (3,1))

print(v.shape == 3)



"""

TODO: usar reshape para v como un vector columna

"""

mat_columna = np.reshape(v, (3, 1))





assert(mat_columna.shape == (3, 1))
A = np.array([

    [1, 2, 3],

    [4, 5, 6],

    [7, 8, 9]

])



B = np.array([

    [1, 1, 1],

    [0, 1, 1],

    [0, 0, 1]

])



""" 

TODO: Calcular el producto interno pedido

"""

prod_interno = A[0] @ B[:, 2]



assert(np.isclose(prod_interno, 6))


u = A[0, :]

v = B[:, 2]



print(u.shape, v.shape)



"""

TODO: Calcular el producto externo

"""

prod_externo = np.reshape(u, (3, 1)) @ np.reshape(v.T, (1, 3))



assert(np.allclose(prod_externo, np.array([

    [1,  1,  1],

    [2,  2,  2],

    [3,  3,  3]]

)))

eps = 1e-6



def tiene_lu(A):

    """

    Dada una matriz A, devuelve True si tiene descomposición LU, o False en caso contrario

    

    Argumentos:

    -----------

    

    A: np.array

        Matriz de floats



    Devuelve:

    ---------

    

    res: bool

        True si tiene LU, False caso contrario

    """

    

    """

    TODO: Completar la función

    """

    res = True



    for i in range(np.shape(A)[0]):

      if np.abs(np.linalg.det(A[:i, :i])) < eps:

        res = False

    return res

    pass 

#Código para que corran los tests: (pueden ignorarlo)



import numpy as np

import math

import traceback





def mntest(func):

    global tests

    

    tests.append(func)    

    

    return func



def correr_tests():

    excepciones = []

    for test in tests:

        try:

            print("Corriendo {} ... ".format(test.__name__), end='')

            test()

            print("OK")

        except AssertionError as e:

            error_msg = traceback.format_exc()

            excepciones.append((test, error_msg))

            print("ERROR")

    

    if len(excepciones) > 0:

        print("\nErrores:\n")

        for (test, error_msg) in excepciones:

            print("En {}".format(test.__name__))

            print(error_msg)

    else:

        print("\n\nTodos los tests pasaron correctamente")
tests = []





@mntest

def testear_identidad_tiene_LU():

    A = np.identity(3)

        

    assert(tiene_lu(A))

    



@mntest

def testear_matriz_ceros_no_tiene_LU():

    A = np.zeros((3, 3))

        

    assert(not tiene_lu(A))

    

@mntest

def testear_matriz_no_inversible():

    A = np.ones((3, 3))

    

    assert(not tiene_lu(A))



@mntest 

def testear_matriz_permutacion_no_tiene_LU():

    A = np.array([

        [1, 0, 0],

        [0, 0, 1],

        [0, 1, 0]

    ])

    

    assert(not tiene_lu(A))





correr_tests()
def lu_en_bloques(A):

    """

    Dada una matriz A, devuelve L, U 

    

    Argumentos:

    -----------

    

    A: np.array

        Matriz de floats



    Devuelve:

    ---------

    

    L, U: np.array

        Descomposición LU de A

        

    """

    

    """

    current_matrix es la matriz a la cual le estoy calculando LU

    

    Primero va a ser A, luego A22 - L21 * U21, ...

    """

    current_matrix = A.copy()

    

    """

    Vamos a ir "rellenando" paso a paso la factorización LU

    """

    L = np.zeros(A.shape)

    U = np.zeros(A.shape)

    

    n = A.shape[0]

    

    """

    Vamos a iterar desde 0 hasta n-1 e ir completando L y U

    de acuerdo a las ecuaciones antes explicadas

    """

    for i in range(n):

        """

        

        Observación: 

        

        En cada paso i estamos "llenando" las columnas y filas i-ésimas de L y U. Por eso tenemos que indexar por i

        

        Sin embargo, current_matrix la tenemos que indexar en 0 ya que es la matriz que vamos a ir

        "achicando" en dimensión. 

        

        """

        

        """

        TODO: Rellenar los valores de L[i, i] y U[i, i]

        """

        L[i, i] = 1

        U[i, i] = current_matrix[0][0]

        """

        Caso "base": si es el final, no seguir

        """

        if i == n-1:

            break

        

        """

        TODO: Calcular los nuevos valores de U12 y L21

        """

        U[i, i+1:] = current_matrix[0, 1:]

        L[i+1:, i] = current_matrix[1:, 0] / U[i][i]

        

        """

        TODO: Calcular la matriz del caso "recursivo".

        

        Esto sería la nueva "A"

        

        Sugerencia: usar np.outer o hacer reshape

        """

        current_dim = current_matrix.shape[0]

        

        new_matrix = current_matrix[1:, 1:] - np.outer(L[i+1:, i] , U[i, i+1:])

        

        """

        Asignamos la nueva matriz a calcular LU

        

        Nos aseguramos de que su dimensión se haya reducido en uno

        """

        current_matrix = new_matrix

        

        assert(current_matrix.shape == (current_dim-1, current_dim-1))



    return L, U
tests = []





@mntest

def testear_con_multiplo_identidad():

    A = 3 * np.identity(3)

    

    L, U = lu_en_bloques(A)

    

    assert(np.allclose(L, np.eye(3)))

    assert(np.allclose(U, 3*np.eye(3)))

    

    



@mntest

def testear_con_otra_matriz():

    L = np.array([

        [1, 0, 0],

        [1, 1, 0],

        [1, 1, 1],

    ])



    U = np.array([

        [1, 1, 1],

        [0, 2, 2],

        [0, 0, 3],

    ])



    A = L @ U

    

    L1, U1 = lu_en_bloques(A)

    

    assert(np.allclose(L1, L))

    assert(np.allclose(U1, U))



@mntest

def testear_con_otra_matriz2():

    A = np.array([

        [8, 2, 0],

        [4, 9, 4],

        [6, 7, 9],

    ])





    

    L1, U1 = lu_en_bloques(A)

    

    assert(np.allclose(L1@U1, A))



    



correr_tests()

def es_ortogonal(A):

    """

    Devuelve True si A es ortogonal, False en otro caso

    

    

    Argumentos:

    ----------

    

    A: np.array

        

    Devuelve:

    ---------

    

    res: bool

        True si A ortogonal, False en otro caso

    """



    """

    TODO: Completar acá

    """

    ret = None



    n = np.shape(A)[0]

    

    At = np.transpose(A)



    Mat_res = A @ At



    Mat_identity = np.eye(n)

    ret = np.allclose(Mat_res, Mat_res)



    return ret
from math import cos, sin

tests = []



@mntest

def probar_con_identidad():

    A = np.eye(3)

    

    assert(es_ortogonal(A))

    

@mntest

def probar_con_rotacion():

    angle = math.pi/4

    """

    TODO: Escribir una matriz de rotación angle de 2x2 

    """

    A = np.array([ [np.cos(angle), -np.sin(angle)],

                   [np.sin(angle), np.cos(angle)]    ])

    

    assert(es_ortogonal(A))



@mntest

def probar_con_reflexion():

    v = np.array([[1,1,1]])

    """

    TODO: Escribir la matriz de reflexión sobre el plano cuya normal es v

    """

    x = np.array([[3**(-1/2),0,0]])



    u = (v-x)

    u = u/np.linalg.norm(u)



    A = np.eye(3) - 2*u*np.transpose(u)

    assert(es_ortogonal(A))



    

correr_tests()
import math as m

def es_simetrica(A):

    for i in range(A.shape[0]):

        for j in range(i,A.shape[1]):

            if not m.isclose(A[i][j],A[j][i]):

                return False

    return True



def no_SDP(diag):

    return np.any(diag < 0)

    

def chol_from_lu(A):

    """

    Devuelve la L de Cholesky a través de la factorización LU de la matriz A

    

    En caso de que no tenga LU o que no tenga Cholesky, lanza ValueError

    

    Argumentos:

    ----------

    

    A: np.array

        Matriz a factorizar

    

    Devuelve:

    ---------

    

    L: np.array

        Factorización de Cholesky de A

    """

    if not es_simetrica(A):

        raise ValueError("Matriz no simétrica")

    

    

    L, U = lu_en_bloques(A)

    

    #Usando 4.1 y 4.2

    

    diag_U = U.diagonal()

    if no_SDP(diag_U):

      raise ValueError("Matriz no SDP")



    D = np.zeros(A.shape)

    #Me armo una matriz D diagonal que tiene la misma diagonal que U

    np.fill_diagonal(D, diag_U)

    D_root = np.sqrt(D)

    L_Cholesky = L @ D_root



    

    return L_Cholesky
tests = []





@mntest

def testear_con_identidad():

    A = np.eye(3)

    L = chol_from_lu(A)

    

    # Cholesky es la identidad también :-)

    assert(np.allclose(A, L))

    



@mntest

def testear_con_multiplo_identidad():

    A = 4 * np.eye(3)

    L = chol_from_lu(A)

    

    # Cholesky es la identidad también :-)

    assert(np.allclose(L, 2* np.eye(3)))

    

@mntest

def testear_con_matriz_no_sdp():

    A = np.array([

        [1, 0, 0],

        [0, -1, 0],

        [0, 0, 2]

    ])

    try:

        L = chol_from_lu(A)

        raise AssertionError("Tiene que lanzar excepción")

    except ValueError as e:

        pass



@mntest

def testear_con_matriz():

    L1 = np.array([

        [1, 0, 0],

        [2, 2, 0],

        [4, 4, 4]

    ])

    

    A = L1 @ L1.T

    L = chol_from_lu(A)

    assert(np.allclose(L, L1))



correr_tests()



def tiene_sdp_vectores_aleatorios(A, n):

    """

    Chequea que la matriz sea SDP usando método probabilístico de vectores aleatorios

    

    Argumentos:

    ----------

    

    A: np.array

        Matriz a verificar su condición de SDP

       

    n: int

        Cantidad de vectores a 

    

    Devuelve:

    ---------

    

    res: bool

        True si no encontró si A es SDP bajo este método probabilístico

    """

    if not es_simetrica(A):

        return False

    

    """

    TODO: Generar vectores aleatorios y verificar si alguno rompe la condición de DP

    """ 

    size = np.shape(A)[0]

    v_random = np.random.normal(3.0, 2.5, size=(size,n))

    sdp = v_random.T.dot(A)

    sdp = sdp.dot(v_random)

    diag = sdp.diagonal()

    return not np.any(diag <= 0)
tests = []





@mntest

def testear_con_identidad():

    A = np.eye(3)

    assert(tiene_sdp_vectores_aleatorios(A, 10000))

    

    

@mntest

def testear_con_matriz_no_sdp():

    A = np.array([

        [1, 0, 0],

        [0, -1, 0],

        [0, 0, 1],

    ])

    

    assert(not tiene_sdp_vectores_aleatorios(A, 10000))

    

    



@mntest

def testear_con_otra_matriz_no_sdp():

    A = np.array([

        [1, 0, 0],

        [0, 1, 2],

        [0, 2, 1],

    ])

    

    assert(not tiene_sdp_vectores_aleatorios(A, 10000))





correr_tests()
