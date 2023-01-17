for i in range(1,11):
    print(i)
suma=0
for i in range (1,11):
    suma=suma+i
    print(i)
print("La suma de los primeros diez numeros naturales es", suma)
entrada=int(input("Introduce un numero entre 2 y 20"))

for i in range(1,entrada+1):
    print("El cubo de",i ,"es",i**3)
    
    
    
    
    

        
        
        
        



numero=int(input("Introduce el numero de la tabla de multiplicar"))
for i in  range(1,11):
    print(i,"x",numero,"=",i*numero)







import random as rn

suma=0
cantidad=10

for i in range(1,cantidad+1):
    ale=rn.randint(1,10) #Genera un numero aleatorio
    print(ale)
    suma+= ale #y lo suma a la variable
print("El promedio de los numeros aleatorios generados es",suma/cantidad)