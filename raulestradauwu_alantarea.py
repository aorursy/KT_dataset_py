import numpy as np

#Busqueda Binaria
def busquedaBinaria (lista, numero_buscado):
    primero = 0
    ultimo = len(lista)-1
    if (numero_buscado>lista[-1]):
        print(f"El numero {numero_buscado} es muy grande para la lista, el mas cercano es: "+ str(lista[-1]))
        return None
    while(primero <= ultimo):
        medio = (primero+ultimo) // 2
        if (lista [medio] == numero_buscado):
            print(numero_buscado,"Fue encontrado en la posición: ", medio)
            break
        elif (lista [medio] < numero_buscado):
            primero = medio + 1
        elif (lista [medio] > numero_buscado):
            ultimo = medio - 1
    if (primero > ultimo):
        print ("Lo sentimos, el numero no fue encontrado en la lista")
        print ("Pero encontramos al numero: "+ str(lista[medio]))
        print ("Y el numero: " + str(lista[medio-+1]))  
    return None
    

Lista=np.random.randint(0,50000,20000)
Lista=sorted(Lista)
print(Lista)

print("¿Que numero desea buscar en la lista?")
busqueda = int(input())
print(busquedaBinaria(Lista , busqueda))
   