lista_1 = [31,23,54,62,48,36,23,57,32,59,37,23,49,34,85,24,23,32,48,29,49,36]
lista_2 = [38,48,58,45,39,23,48,58,28,49,64,49,38,28,49,29,39,15,39,29,49,28]
lista_3 = [99,54,84,65,54,65,51,45,75,98,52,35,24,36,44,45,60,56,35,29,32,66]
5+3 
# Obtener la longitud de la lista_1 
N_ = len(lista_1) 
# Inicializar las variables de interes
max_val = min_val = mean_val = lista_1[0]
# Iterar a traves de la lista para obtener los valores
for index in range(1,N_):
    # Indicar el numero de prueba
    test_val = lista_1[index]
    # Revisar si es mayor al valor maximo actual
    if test_val > max_val:
        # Si es asi, asignar el nuevo valor maximo
        max_val = test_val
    # Revisar si es menor al valor minimo actual 
    if test_val < min_val:
        # Si es asi, asignal el nuevo valor minimo
        min_val = test_val
    # Agregar el valor actual al valor acumulado
    mean_val += test_val
# Generar el promedio de valores en la lista
mean_val = mean_val / N_

print("Valores encontrados de la lista_1 \n\n\t max : {} \t min : {} \t mean : {} \n".format(max_val, min_val, mean_val))
def revisar_lista(lista):
    # Obtener la longitud de la lista 
    N_ = len(lista) 
    # Inicializar las variables de interes
    max_val = min_val = mean_val = lista[0]
    # Iterar a traves de la lista para obtener los valores
    for index in range(1,N_):
        # Indicar el numero de prueba
        test_val = lista[index]
        # Revisar si es mayor al valor maximo actual
        if test_val > max_val:
            # Si es asi, asignar el nuevo valor maximo
            max_val = test_val
        # Revisar si es menor al valor minimo actual 
        if test_val < min_val:
            # Si es asi, asignal el nuevo valor minimo
            min_val = test_val
        # Agregar el valor actual al valor acumulado
        mean_val += test_val
    # Generar el promedio de valores en la lista
    mean_val = mean_val / N_
    
    return [max_val, min_val, mean_val]
[max_val_1, min_val_1, mean_val_1] = revisar_lista(lista_1)
[max_val_2, min_val_2, mean_val_2] = revisar_lista(lista_2)
[max_val_3, min_val_3, mean_val_3] = revisar_lista(lista_3)