# Probabilidades de éxito de la expansión por país
prob_exito = {'Australia': [0.6, 0.33, 0.11, 0.14],
              'Francia': [0.66, 0.78, 0.98, 0.2],
              'Italia': [0.6],
              'Brasil': [0.22, 0.22, 0.43],
              'USA': [0.2, 0.5, 0.3],
              'Inglaterra': [0.45],
              'Canadá': [0.25, 0.3],
              'Argentina': [0.22],
              'Grecia': [0.45, 0.66, 0.75, 0.99, 0.15, 0.66],
              'Marruecos': [0.29],
              'Túnez': [0.68, 0.56],
              'Egipto': [0.99],
              'Jamaica': [0.61, 0.65, 0.71],
              'Suiza': [0.73, 0.86, 0.84, 0.51, 0.99],
              'Alemania': [0.45, 0.49, 0.36]}
print(prob_exito)
# Vistazo a las llaves del diccionario
list(prob_exito.keys())
# ... y sus valores respectivamente
list(prob_exito.values())
print('Verificar si la llave de Marruecos existe:')
print('Marruecos' in prob_exito)

print('Verificar si la llave de Japón existe:')
print('Japón' in prob_exito)
prob_exito['Jamaica'] 
lista_jamaica = prob_exito['Jamaica'] 
print(lista_jamaica)
# Cada print imprimirá su resultado en una nueva línea
print(lista_jamaica[0]) # imprime el primer elemento
print(lista_jamaica[1]) # imprime el segundo elemento
print(lista_jamaica[2]) # imprime el tercer elemento
print(lista_jamaica[-1]) # imprime el último elemento
print(lista_jamaica[-2]) # imprime el penúltimo elemento
print(lista_jamaica[-3]) # imprime el antepenúltimo elemento
len(lista_jamaica) # devuelve el tamaño de la lista
print('Cantidad de estimaciones para Francia:')
print(len(prob_exito['Francia']))

print('Cantidad de estimaciones para Grecia:')
print(len(prob_exito['Grecia']))

print('Cantidad de estimaciones para Marruecos:')
print(len(prob_exito['Marruecos']))
avg_jamaica = (0.61 + 0.65 + 0.71) / 3
print(avg_jamaica)
nombre_pais = 'Jamaica'
lista_jamaica = prob_exito[nombre_pais] # lista de probabilidades para Jamaica
print(lista_jamaica)
avg_jamaica = sum(lista_jamaica) / len(lista_jamaica)
min_jamaica = min(lista_jamaica)
max_jamaica = max(lista_jamaica)
print("País:",nombre_pais,", Promedio:",avg_jamaica)
print("País:",nombre_pais,", Mínimo:",min_jamaica)
print("País:",nombre_pais,", Máximo:",max_jamaica)
avg_jamaica = round(sum(lista_jamaica) / len(lista_jamaica),2)
min_jamaica = round(min(lista_jamaica),2)
max_jamaica = round(max(lista_jamaica),2)
print("País:",nombre_pais,", Promedio:",avg_jamaica)
print("País:",nombre_pais,", Mínimo:",min_jamaica)
print("País:",nombre_pais,", Máximo:",max_jamaica)
# obtener todas las llaves del diccionario prob_exito
lista_nombre_paises = list(prob_exito.keys())
print(lista_nombre_paises)
# Hacer un cico sobre todos los países y calcular su probabilidad promedio de éxito
for i in lista_nombre_paises:
    print('--Inicio de la iteración--')
    print('Elemento de lista_nombre_paises en la variable i = ' + i)
    print('Valores del diccionario en prob_exito[i]: ', prob_exito[i])
    print('Promedio de prob_exito[i]: ', sum(prob_exito[i]) / len(prob_exito[i]))
    print('--Fin de la iteración--')
for i in lista_nombre_paises:
    print('País: ',i,', Min: ', min(prob_exito[i]))
    print('País: ',i,', Max: ', max(prob_exito[i]))
for i in lista_nombre_paises:
    rango_pais = max(prob_exito[i]) - min(prob_exito[i])
    print('País: ', i, ", Rango: ", rango_pais)
    
# Analizando los resultados vemos que Grecia tiene el rango más alto.
# Toma cada elemento i en prob_exito y pone i en lista_llaves
lista_llaves = [i for i in prob_exito]
lista_llaves
# Toma cada elemento i en prob_exito y pone prob_exito[i] en lista_llaves
lista_valores = [prob_exito[i] for i in prob_exito]
lista_valores
# Cantidad de probabilidades por país
[[i,len(prob_exito[i])] for i in prob_exito]
# Solución posible
lista_suma_cuadrados = [[i, sum([j**2 for j in prob_exito[i]])] for i in prob_exito]
lista_suma_cuadrados
# Posible solución
lista_menos_promedio = [[i, [round(j - sum(prob_exito[i])/len(prob_exito[i]),2) for j in prob_exito[i]]] for i in prob_exito]
lista_menos_promedio
# Obtener lista  países
lista_nombre_paises = list(prob_exito.keys())

# Crear un diccionario vacío, que contendrá las probabilidades medias por país
promedio_paises = {}

# Ciclo sobre la lista de países y cálculo de probabilidad promedio
for i in lista_nombre_paises:
    lista_prob_pais = prob_exito[i] # lista de probabilidades para cada país

    # Si el país tiene más de una probabilidad estimada, guarde el registro, si no, pase al siguiente país
    if len(prob_exito[i]) > 1:
        promedio_pais = sum(lista_prob_pais) / len(lista_prob_pais)
        promedio_paises[i] =  promedio_pais # insertar el promedio en el diccionario usando el país como llave
# Nicely format the result for printing to the screen
for llave_pais in promedio_paises: 
    print("País: {0:s}, Estimación promedio: {1:.2f}".format(llave_pais, promedio_paises[llave_pais]))
# Crear lista de países
lista_nombre_paises = list(prob_exito.keys())

# Crear un diccionario vacío, que contendrá las probabilidades medias por país
promedio_paises = {}

# Ciclo sobre la lista de países y cálculo de probabilidad promedio
for i in lista_nombre_paises:
    lista_prob_pais = prob_exito[i] # list of estimates for a country

    # Si el país tiene más de tres probabilidades estimadas, guarde el registro, si no, pase al siguiente país
    if len(prob_exito[i]) > 2:
        promedio_pais = sum(lista_prob_pais) / len(lista_prob_pais)
        promedio_paises[i] =  promedio_pais # insertar el promedio en el diccionario usando el país como llave
        print("País: {0:s}, Estimación promedio: {1:.2f}".format(i, promedio_pais))
    else:
        print("País: {0:s}, *No cumple la política de la empresa*".format(i))
lista_nombre_paises = list(prob_exito.keys())
for i in lista_nombre_paises:
    min_est = min(prob_exito[i])
    prom_est = sum(prob_exito[i]) / len(prob_exito[i])
    max_est = max(prob_exito[i])
    largo_est = len(prob_exito[i])
    cumple_politica = largo_est > 2
    print('País:',i,', Mínimo:',min_est,', Promedio:',prom_est,', Máximo:',max_est,', Estimaciones:',largo_est,', Cumple Política:',cumple_politica)
