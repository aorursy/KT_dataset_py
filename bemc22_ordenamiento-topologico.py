import math

import matplotlib.pyplot as plt

import networkx as nx
# Cada indice del diccionario es un vertices y sus items son aquellos vertices que son adyacentes a este

# El grafo que usaremos debe ser un grafo aciclico dada la definicion del algoritmo



grafo = {

    6 : {4,5},

    5 : {},

    4 : {5},

    3 : {5},

    2 : {3 , 4},

    1 : {2 , 3 , 6}

}



# muestra [1, 2, 3, 4, 6, 5]
# Cada indice del diccionario es un vertices y sus items son aquellos vertices que son adyacentes a este

# El grafo que usaremos debe ser un grafo aciclico dada la definicion del algoritmo

grafo = {

    1 : {2 , 3 , 6},

    2 : {3 , 4},

    3 : {5},

    4 : {5},

    5 : {},

    6 : {4,5}

}

# muestra  [1, 6, 2, 4, 3, 5]
def recorrer(aristas,padre,grafo):

    if(len(aristas) == 0):

        return 0         # Si el vertice en el que iteramos no tiene vertices adyacentes retornamos 0

    for i in aristas:    # De lo contrario recorremos todos los vertices adyacentes

        if(i == padre):  # Si algun nodo es el nodo padre retornamos 1 

            return 1     # Esto significa que el grafo contiene ciclos

        

    # Ahora si ninguno de los dos casos anteriores se cumple entonces 

    # Iteramos recursivamente en los nodos adyacentes a los nodos adyacentes actuales

    for v in aristas:

        return  0 + recorrer(grafo[v],padre,grafo)



def DAG(grafo):

    x = 0 # Contandor inicia en 0: El grado es ciclico

    for i in grafo: # Iteramos en cada uno de los vertices de nuestro grafo

        x += recorrer(grafo[i],i,grafo)  

    if(x != 0):

        print("El grafo contiene ciclos")

    else:

        print("El grafo es acíclico")

        
def visitar(u,estado,lista):

    # Al entrar a la funcion visitar()

    # Significa que estamos visitando el vertices u 

    # luego lo guardamos como True: Visitado

    # en nuestro diccionario estado

    estado[u] = True

    for v in grafo[u]: # Recorremos todos los vertices adyacentes al vertice u

        if(estado[v] == False):

            # Si aun no hemos visitado el vertice v

            # entonces lo visitamos

            visitar(v,estado,lista)

            

    # Luego de haber visitado todos los vertices adyacentes de u

    # ya no tenemos que seguir utilizando a u entonces

    lista.insert(0,u) # Insertamos el vertice u al inicio de nuestra lista
def ordenamiento_topologico(grafo, lista):

    # Almacena el estado de cada vertice, False:No Visitado, True: Visitado

    estado = {} 

    for u in grafo:

        # Al inicio agregamos todos los vertices a nuestro diccionario y

        # y guardamos su estado como False: No visitado

        estado[u] = False

    for u in grafo: # Recorremos todos los vertices del grafo

        if( estado[u] == False):

            # Si aun no hemos visitado el vertice u

            # entonces lo visitamos

            visitar(u,estado,lista)
# Creacion del grafo

G = nx.DiGraph()



# Agregamos las aristas al grafo

# Al agregar las aristas implicitamente

# Se agregan los nodos

for i in grafo:

    x = grafo[i]

    for j in x:

        G.add_edge(i,j)

    

# Diccionario con las posiciones de cada vertice

dict = {

    1 : (0,1),

    2 : (0.8,1),

    3 : (0.8,0),

    4 : (1.6,1),

    5 : (2,0),

    6 : (2,2)

}



pos = dict
# Tamaño del canvas

plt.figure(figsize = (8,8))

# Se dibujan las aristas

nx.draw_networkx_edges(G, pos, node_size=1800, width=2, arrowsize=25)

# Se dibujan los nodos

nx.draw_networkx_nodes(G, pos, node_size=1800, node_color =  (0, 0.6, 1))

# Se dibujan las etiquetas

nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis("off") # Con esta linea ocultamos los ejes de coordenadas

plt.show()      # Mostramos todas las llamadas de dibujado realizadas anteriormente
DAG(grafo)
lista = []

ordenamiento_topologico(grafo,lista)

print("El ordenamiento topologico del grafo es: ",lista)
# Creamos el nuevo grafo que ahora usaremos

grafo = {

    1  : {2,3,6},

    2  : {3,4,6},

    3  : {5,6},

    4  : {6,7},

    5  : {4,6,8},

    6  : {7,8,9},

    7  : {9,10},

    8  : {9,10},

    9  : {10},

    10 : {}

}



# muestra [1, 2, 3, 5, 4, 6, 7, 8, 9, 10]
# Creacion del grafo

G = nx.DiGraph()





# Agregamos las aristas al grafo

# Al agregar las aristas implicitamente

# Se agregan los nodos

for i in grafo:

    x = grafo[i]

    for j in x:

        G.add_edge(i,j)

        

# Diccionario con las posiciones de cada vertice        

dict = {

    1  : (1,11),

    2  : (0,9),

    3  : (2,9),

    4  : (0,6),

    5  : (2,6),

    6  : (1,4.5),

    7  : (0,3),

    8  : (2,3),

    9  : (1,2),

    10 : (1,0) 

    

}



pos = dict
# Tamaño del canvas

plt.figure(figsize = (5,10))

# Se dibujan las aristas

nx.draw_networkx_edges(G, pos, node_size=1800, width=2, arrowsize=25)

# Se dibujan los nodos

nx.draw_networkx_nodes(G, pos, node_size=1800, node_color =  (0, 0.6, 1))

# Se dibujan las etiquetas

nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis("off") # Con esta linea ocultamos los ejes de coordenadas

plt.show()      # Mostramos todas las llamadas de dibujado realizadas anteriormente
DAG(grafo)
lista = []

ordenamiento_topologico(grafo,lista)

print("El ordenamiento topologico del grafo es: ",lista)
grafo = {

    1  : {2,3,4,5},

    2  : {8,9},

    3  : {5,6,9},

    4  : {3,7,9},

    5  : {9,11,14},

    6  : {9,13,12,16},

    7  : {6,13,15},

    8  : {9,10,11,12},

    9  : {13},

    10 : {14,15},

    11 : {10,12,15,14,17},

    12 : {16,17},

    14 : {},

    13 : {16,18},

    15 : {17,18},

    16 : {18,19},

    17 : {18,20},

    18 : {20,19},

    19 : {},

    20 : {}

}



# muestra [1, 4, 7, 3, 6, 5, 2, 8, 11, 12, 10, 15, 17, 14, 9, 13, 16, 18, 20, 19]
# Creacion del grafo

G = nx.DiGraph()





# Agregamos las aristas al grafo

# Al agregar las aristas implicitamente

# Se agregan los nodos

for i in grafo:

    x = grafo[i]

    for j in x:

        G.add_edge(i,j)



# Diccionario con las posiciones de cada vertice

dict = {

    1  : (3,7),

    2  : (2,6),

    3  : (3,6),

    4  : (4,6),

    5  : (2,5),

    6  : (4,5),

    7  : (5,5),

    8  : (1,4),

    9  : (3,4),

    10 : (0,3),

    11 : (2,3),

    12 : (3,3),

    13 : (5,3),

    14 : (1,2),

    15 : (3,2),

    16 : (4,2),

    17 : (2,1),

    18 : (4,1),

    19 : (5.5,1),

    20 : (3,0)

    

    

}



pos = dict
# Tamaño del canvas

plt.figure(figsize = (18,18))

# Se dibujan las aristas

nx.draw_networkx_edges(G, pos, node_size=1800, width=2, arrowsize=25)

# Se dibujan los nodos

nx.draw_networkx_nodes(G, pos, node_size=1800, node_color =  (0, 0.6, 1))

# Se dibujan las etiquetas

nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis("off") # Con esta linea ocultamos los ejes de coordenadas

plt.show()      # Mostramos todas las llamadas de dibujado realizadas anteriormente
DAG(grafo)
lista = []

ordenamiento_topologico(grafo,lista)

print("El ordenamiento topologico del grafo es: ",lista)