def comparaListas(lista1, lista2):

    if lista1 == lista2:

        return True

    

    return False

    

def comparaElementosDaLista(lista1, lista2):

    l1 = set(lista1)

    l2 = set(lista2)

    

    if(len(l1.symmetric_difference(l2)) == 0):

        return True

    

    return False
a = [1,3,"a",4,5,6,7]

b = [1,3,"a",4,5,7,6,7,6]

print(comparaListas(a,b))

print(comparaElementosDaLista(a,b))