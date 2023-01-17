### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 1 linea de código)

test = "Hola Mundo"

### TERMINE EL CÓDIGO AQUÍ ###
print ("test: " + test)
# FUNCIÓN A CALIFICAR basic_sigmoid



import math



def basic_sigmoid(x):

    """

    Calcula el sigmoide de x

    Input:

    x: scalar

    Output:

    s: sigmoid(x)

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 1 linea de código)

    s = 1 / (1 + math.exp(-x))

    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return s
basic_sigmoid(3)
### Una razón para utilizar "numpy" en lugar de "math" en Deep Learning ###

x = [1, 2, 3]

basic_sigmoid(x) # esto da error, porque x es un vector
import numpy as np



# ejemplo de np.exp

x = np.array([1, 2, 3])

print(np.exp(x)) 
# ejemplo de una operación vectorial

x = np.array([1, 2, 3])

print (x + 3)
# FUNCIÓN A CALIFICAR: sigmoid



import numpy as np # esto permite acceder funciones numpy simplemente escribiendo np.function() en lugar de numpy.function()



def sigmoid(x):

    """

    Calcule el sigmoide de x

    Input:

    x: un escalar o arreglo numpy de cualquier tamaño

    Output:

    s: sigmoid(x)

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 1 linea de código)

    s = 1 / (1 + np.exp(-x))

    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return s
x = np.array([1, 2, 3])

sigmoid(x)
# FUNCIÓN A CALIFICAR: sigmoid_derivative



def sigmoid_derivative(x):

    """

    Calcule el gradiente (o derivada) de la función sigmoide con respecto al input x.

    Puede guardar el output del sigmoide como variables y luego usarlo para calcular el gradiente.

    Input:

    x: un escalar o arrgelo numpy 

    Output:

    ds: el gradiente calculado.

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 2 lineas de codigo)

    s = sigmoid(x)

    ds = s * (1 - s)

    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return ds
x = np.array([1, 2, 3])

print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
# FUNCIÓN A CALIFICAR: image2vector



def image2vector(image):

    """

    Input:

    image: un arreglo numpy con forma (longitud, altura, profundidad)

    Output:

    v: un vector con forma (longitud*altura*profundidad, 1)

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 1 linea de código)

    v = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return v
# Este es un arreglo de 3 por 3 por 2, usualmente las imagenes son de (num_px_x, num_px_y,3) donde 3 representa los valores RGB

image = np.array([[[ 0.67826139,  0.29380381],

        [ 0.90714982,  0.52835647],

        [ 0.4215251 ,  0.45017551]],



       [[ 0.92814219,  0.96677647],

        [ 0.85304703,  0.52351845],

        [ 0.19981397,  0.27417313]],



       [[ 0.60659855,  0.00533165],

        [ 0.10820313,  0.49978937],

        [ 0.34144279,  0.94630077]]])



print ("image2vector(image) = " + str(image2vector(image)))
# FUNCIÓN A CALIFICAR: normalizeRows



def normalizeRows(x):

    """

    Implemente una función que normalize cada fila de la matriz x (para que tenga longitud unitaria).

    Input:

    x: Un arreglo numpy con forma (n, m)

    Output:

    x: La matriz numpy normalizada por filas. 

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 2 lineas de código)

    # Compute x_norm como la norma 2 de x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)

    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    

    # Divida a x por su norma.

    x = x / x_norm

    ### TERMINE EL CÓDIGO AQUÍ ###



    return x
x = np.array([

    [0, 3, 4],

    [1, 6, 4]])

print("normalizeRows(x) = " + str(normalizeRows(x)))
# FUNCIÓN A CALIFICAR: softmax



def softmax(x):

    """

    Calcule el softmax para cada fila del input x.

    El código debe funcionar tanto para un vector fila como para matrices de tamaño (n, m).

    Input:

    x: un arreglo numpy con forma (n,m)

    Output:

    s: Una matriz numpy igual al softmax de x, de tamaño (n,m)

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 3 lineas de código)

    # Utilize exp() sobre cada elemento de x. Use np.exp(...).

    x_exp = np.exp(x)



    # Defina el vector x_sum que sume cada fila de x_exp. Use np.sum(..., axis = 1, keepdims = True).

    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    

    # Compute softmax(x) dividiendo x_exp por x_sum. Debería usar automáticamente numpy broadcasting.

    s = x_exp / x_sum



    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return s
x = np.array([

    [9, 2, 5, 0, 0],

    [7, 5, 0, 0 ,0]])

print("softmax(x) = " + str(softmax(x)))
import time



x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]

x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]



### IMPLEMENTACION CLASICA DEL PRODUCTO PUNTO ENTRE DOS VECTORES ###

tic = time.process_time()

dot = 0

for i in range(len(x1)):

    dot+= x1[i]*x2[i]

toc = time.process_time()

print ("interno = " + str(dot) + "\n ----- Tiempo computacional = " + str(1000*(toc - tic)) + "ms")



### IMPLEMENTACION CLÁSICA DEL PRODUCTO EXTERIOR ###

tic = time.process_time()

outer = np.zeros((len(x1),len(x2))) # matriz de ceros de tamaño len(x1)*len(x2)

for i in range(len(x1)):

    for j in range(len(x2)):

        outer[i,j] = x1[i]*x2[j]

toc = time.process_time()

print ("externo = " + str(outer) + "\n ----- Tiempo computacional = " + str(1000*(toc - tic)) + "ms")



### IMPLEMENTACION CLÁSICA POR ELEMENTOS ###

tic = time.process_time()

mul = np.zeros(len(x1))

for i in range(len(x1)):

    mul[i] = x1[i]*x2[i]

toc = time.process_time()

print ("multiplicación por elementos = " + str(mul) + "\n ----- Tiempo computacional = " + str(1000*(toc - tic)) + "ms")



### IMPLEMENTACION CLASICA GENERAL DEL PRODUCTO PUNTO ###

W = np.random.rand(3,len(x1)) # Arreglo numpy aleatorio de tamaño 3*len(x1) 

tic = time.process_time()

gdot = np.zeros(W.shape[0])

for i in range(W.shape[0]):

    for j in range(len(x1)):

        gdot[i] += W[i,j]*x1[j]

toc = time.process_time()

print ("g_interno = " + str(gdot) + "\n ----- Tiempo computacional = " + str(1000*(toc - tic)) + "ms")
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]

x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]



### PRODUCTO INTERNO VECTORIZADO ###

tic = time.process_time()

dot = np.dot(x1,x2)

toc = time.process_time()

print ("interno = " + str(dot) + "\n ----- Tiempo computacional =  " + str(1000*(toc - tic)) + "ms")



### PRODUCTO EXTERNO VECTORIZADO ###

tic = time.process_time()

outer = np.outer(x1,x2)

toc = time.process_time()

print ("externo = " + str(outer) + "\n ----- Tiempo computacional =  " + str(1000*(toc - tic)) + "ms")



### MULTIPLICACION POR ELEMENTOS VECTORIZADA ###

tic = time.process_time()

mul = np.multiply(x1,x2)

toc = time.process_time()

print ("multiplicación por elementos = " + str(mul) + "\n ----- Tiempo computacional =  " + str(1000*(toc - tic)) + "ms")



### PRODUCTO INTERNO GENERAL VECTORIZADO ###

tic = time.process_time()

dot = np.dot(W,x1)

toc = time.process_time()

print ("g_interno = " + str(dot) + "\n ----- Tiempo computacional =  " + str(1000*(toc - tic)) + "ms")
# FUNCIÓN A CALIFICAR: L1



def L1(yhat, y):

    """

    Input:

    yhat: vector de tamaño m (etiquetas estimadas)

    y: vector de tamaño m (etiquetas observadas)

    Output:

    loss: el valor de la pérdida L1 definida arriba

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 1 linea de código)

    loss = np.sum(np.abs(y - yhat))

    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

print("L1 = " + str(L1(yhat,y)))
# FUNCIÓN A CALIFICAR: L2



def L2(yhat, y):

    """

    Input:

    yhat: vector de tamaño m (etiquetas estimadas)

    y: vector de tamaño m (etiquetas observadas)

    Output:

    loss: el valor de la pérdida L2 definida arriba

    """

    

    ### EMPIEZE EL CÓDIGO AQUÍ ### (≈ 1 linea de código)

    loss = np.sum(np.dot(y - yhat, y - yhat))

    ### TERMINE EL CÓDIGO AQUÍ ###

    

    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])

y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2(yhat,y)))