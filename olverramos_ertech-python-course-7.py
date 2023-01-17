import matplotlib.pyplot as plt # Biblioteca para graficar
#instrucción para graficar de manera directa
%matplotlib inline  
def es_par(numero):
    if número % 2 == 0:
        return True
    else:
        return False
def es_par(numero):
    pasos = 0
    pasos += 1
    if numero % 2 == 0:
        pasos += 1
        return True, pasos
    else:
        pasos += 1
        return False, pasos
def graficar (lista_n, funcion):
    cantidad_pasos = []
    for n in lista_n:
        resultado, pasos = funcion(n)
        cantidad_pasos.append(pasos)
    print (lista_n)
    print (cantidad_pasos)
    fig, ax = plt.subplots()
    ax.plot(lista_n, cantidad_pasos)
    ax.set(xlabel='n', ylabel='Cantidad Pasos',
           title='Cantidad de Pasos Vs n')
    ax.grid()
lista_n = (0,1,10,50,100,500,1000,3000,10000,100000)
graficar (lista_n, es_par)
def es_primo(n):
    resultado = True 
    if n < 2:
        resultado = False
    i = 2
    while i < n:
        if n % i == 0:
            resultado = False
        i += 1
    return resultado
es_primo(10)
def es_primo(n):
    pasos = 0
    pasos += 1
    resultado = True
    pasos += 1
    if n < 2:
        pasos += 1
        resultado = False
    pasos += 1
    i = 2
    pasos += 1
    while i < n:
        pasos += 1
        if n % i == 0:
            pasos += 1
            resultado = False
        pasos += 1
        i += 1
    pasos += 1        
    return resultado, pasos
n = 10
es_primo(n)
2 * n + 1 + 2 
lista_n = (0, 1) #(0,1,10,23,50,85,87,100,500,1000,3000,10000,100000)
graficar (lista_n, es_primo)
# R/
def ordernar(L):
    L2 = []
    ########
    # Algorítmo para ordenar la lista 
    ########
    
    return L2
# R/
def ordernar(L):
    pasos = 0
    L2 = []
    ########
    # Algorítmo para ordenar la lista 
    ########
    pass
    # Debe retornar la cantidad de pasos
    return L2, pasos

def generar_lista_aleatoria(n):
    L = []
    ####
    # Algorítmo para generar una lista aleatoria con n números enteros entre 0 y n-1.
    ####
    return L

def ordenar_lista_aleatoria(n):
    L = generar_lista_aleatoria(n)
    return ordernar(L)

# R/
import random
random.seed(20200620)

lista_n = (0,1) #(0,1,10,23,50,85,87,100,500,1000,3000,10000,100000)
graficar (lista_n, ordenar_lista_aleatoria)

# R/
def buscar(L, x):
    indice = None
    ####
    # Algoritmo
    ####
    return indice
# R/
def buscar(L, x):
    pasos = 0
    pasos += 1
    indice = None
    ####
    # Algoritmo
    ####
    pasos += 1
    return indice, pasos
L = [ 6, 4, 7, 2, 5 ]
buscar (L, 9)
def buscar_lista_aleatoria(n):
    L = generar_lista_aleatoria(n)
    x = len(L) + 1
    return buscar(L, x)
lista_n = (0,1) #(0,1,10,23,50,85,87,100,500,1000,3000,10000,100000)
graficar (lista_n, buscar_lista_aleatoria)
# R/
def buscar_ordenada(L, x):
    indice = None
    ####
    # Algoritmo
    ####
    return indice

# Con conteo de pasos
def buscar_ordenada(L, x):
    pasos = 0
    pasos += 1
    indice = None
    ####
    # Algoritmo
    ####
    pasos += 1
    return indice, pasos
L = [ 2, 4, 5, 6, 7 ]
buscar_ordenada (L, 9)
def buscar_ordenada_lista_aleatoria(n):
    L = generar_lista_aleatoria(n)
    L, pasos = ordernar(L)
    x = len(L) + 1
    indice, pasos_busqueda = buscar_ordenada(L, x)
    return indice, pasos_busqueda 

lista_n = (0,1) #(0,1,10,23,50,85,87,100,500,1000,3000,10000,100000)
graficar (lista_n, buscar_ordenada_lista_aleatoria)