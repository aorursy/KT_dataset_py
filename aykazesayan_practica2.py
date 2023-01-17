import csv as csv
import numpy as np
import random

#Puntuaciones

MUJER = 40
HOMBRE = 15

CLASE1 = 20
CLASE2 = 10
CLASE3 = 5

JOVENES = 25							# Menor de 20
ADULTOS = 17							# Entre 20 y 50
ANCIANOS = 7							# Mayor de 50


# First, read in test.csv
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

result_file = open("result.csv", "wb")
result_file_object = csv.writer(result_file)
result_file_object.writerow(["PassengerId", "Survived"])	


def puntuacionClase(row, puntuacion):
	if row[1] == '1':
		puntuacion = puntuacion + CLASE1
	elif row[1] == '2':
		puntuacion = puntuacion + CLASE2
	else:
		puntuacion = puntuacion + CLASE3
		 
	return puntuacion


def puntuacionSexo(row, puntuacion):				
    if row[3] == 'female':										
        puntuacion = puntuacion + MUJER		
    else:														
        puntuacion = puntuacion + HOMBRE
        
    return puntuacion
    
def puntuacionEdad(row, puntuacion):				
	if row[4] < 20:					
		puntuacion = puntuacion + JOVENES
	elif row[4] >= 20 and row[4] < 50:
		puntuacion = puntuacion + ADULTOS
	else:
		puntuacion = puntuacion + ANCIANOS
	return puntuacion
    

for row in test_file_object:
	
	puntuacion = 0
	puntosSupervivencia = random.randint(0, 100)
	
	


	puntuacion = puntuacionClase(row, puntuacion)
	puntuacion = puntuacionSexo(row, puntuacion)
	puntuacion = puntuacionEdad(row, puntuacion)

	if puntuacion >= puntosSupervivencia:
		result_file_object.writerow([row[0], "1"])
	else:
		result_file_object.writerow([row[0], "0"])
		
	
test_file.close()												
result_file.close()









