import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
#se hace la carga del log, se utiliza el caracter de espacio 

#para separar las columnas y no se ponen encabezados



ez = pd.read_csv("/kaggle/input/ezproxy-log/ezp201805.log", sep=" ", header=None)
# la instrucción shape permite ver la 

#cantidad de registros y la cantidad de columnas



ez.shape
# la instrucción head permite ver los primeros 5 registros del log



ez.head()
# se realiza una búsqueda en todos los registros teniendo en cuenta 

#la cadena de caracteres que queremos para filtrar una base de datos



# str.contains permite encontrar una cadena de texto en una transacción



# # La opción case habilita o deshabilita la búsqueda en respetando mayúsculas y minúsculas 

# # La opción na habilita o deshabilita la opción de incluir cadenas vacías

# # La opción regex habilita o deshabilita la opción de expresiones regulares



cadena1 = "www.clinicalkey.es"



sesiones = ez[ez[5].str.contains(cadena1, case=False, na=False, regex=False)] 



# la instrucción print permite imprimir el resultado en pantalla

# se imprimen la cantidad de sesiones únicas de la variable sessions

# la instrucción len permite identificar la cantidad de registros en una columna



print("Cantidad de sesiones: ", len(sesiones[2].unique()))
# se realiza una búsqueda en "bd" teniendo en cuenta la cadena de caracteres que queremos

# str.contains permite encontrar una cadena de texto en una transacción

# # La opción case habilita o deshabilita la búsqueda en respetando mayúsculas y minúsculas 

# # La opción na habilita o deshabilita la opción de incluir cadenas vacías

# # La opción regex habilita o deshabilita la opción de expresiones regulares



cadena2 = "/ui/service/search"



busquedas = sesiones[sesiones[5].str.contains(cadena2, case=False, na=False, regex=False)] 



# la instrucción print permite imprimir el resultado en pantalla

# se imprimen la cantidad de sesiones realizadas en "bd"

# la instrucción len permite identificar la cantidad de registros en una columna



print("Cantidad de búsquedas: ", len(busquedas[5]))
# una vez identificadas las sesiones y las búsquedas, descubriremos como se comportan 

# las búsquedas en el periodo que contiene el LOG



# las fechas estás de la siguiente manera en la columna 4:   [01/May/2020:00:00:01

# lo que queremos es dejar únicamente el mes, el día y el año. Por lo que descartarémos la hora

# se utiliza str.split para indicar que se separará la columna 4 con el limitador dos puntos (:)

# # La opción expand permité dividir y quitar la información que ya no queremos de la columna 4



busquedas[3] = busquedas[3].str.split(":", expand=True)



# una vez realizado esto, la fecha quedará de la siguiente forma [01/May/2020

# quitarémos el caracter [ del inicio de la fecha

# se utiliza str.replace para reemplazar [ por nada



busquedas[3] = busquedas[3].str.replace("[", "")



# se utiliza la instrucción hist para crear un histograma con las fechas que contienen búsquedas

# figsize permite modificar el tamaño de la gráfica resultante



busquedas[3].hist(figsize=(40,5), bins=24)