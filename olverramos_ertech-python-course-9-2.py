class Tree:
    ##TODO: Implementar en clase paso a paso
    pass


def buscar_recursivo(L, x):
    if len(L) == 0:
        return None
    print (f"Buscar {x} en {L}")
    i = len(L) // 2
    if L[i] == x:
        return i
    elif x < L[i]:
        return buscar_recursivo(L[:i], x)
    else:
        temporal = buscar_recursivo(L[i+1:], x)
        if temporal is None:
            return None
        return i + 1 + temporal
L = [4, 6, 8, 12, 18, 25, 34, 48, 52, 78]
x = 17
result = buscar_recursivo(L, x)
if result is None:
    print (f"No se encontró el {x}")
else:
    print (f"Se encontró el {x} en la posición {result}")