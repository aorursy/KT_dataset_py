##Almacenamos en una variable un valor

numero=10

##evaluamos con la condicional

if numero > 9: #Evaluacion de la condicion

    print("El número es mayor a 8") #Sentencia si se cumple la condición

    print("Esto esta dentro del IF")

    

print("Esto no esta dentro del IF")

    

##### 

numero=5

if numero > 5:

    print("El numero es mayor a 5")

else:

    print("El numero no es mayor a 5")
##### 

numero=5

if numero > 5:

    print("El numero es mayor a 5")

elif numero < 5:

    print("El numero es menor a 5")

else:

    print("el numero es igual a 5")
#Comprobar si un numero esta en el rango desde 10 hasta 20

numero=15



#Manera tradicional

if numero >=10:

    if numero <= 20:

        print("El numero",numero ,"si esta en el rango de 10 a 20")



#Uso de operadores logicos AND

if numero >= 10 and numero <= 20:

    print("El numero",numero ,"si esta en el rango de 10 a 20")



#Uso de OR

#Evaluar si un numero es mayor que 20 o menor que 10, es decir, lo que esta fuera del rango 10 a 20.

numero=21

if numero >20 or numero <10:

    print("el numero",numero, "Esta fuera de rango")




