contador = 1

#range(3)--> [0,1,2]

for contador in range(3):

    print(contador)
# Muestra todos los números entre 0,1,2,3,4

for x in range(5):

    print(x)



# Muestra 3,4,5

for x in range(3,6):

    print(x)

    

nombres=["Juan","Maria","Jose"]

for name in nombres:

    print(name)



lista=[6,9,3,7,3,8]

for i in lista: #Acceso directo a los elementos de la lista

    print(i)



print("")

for i in range(6): #Acceso mediante referencia de posicion

    print(lista[i])
count = 0

while count < 5:

    print(count)

    count = count + 1
# Muestra 0,1,2,3,4



count = 0

while True:

    print(count)

    count += 1

    if count >= 5:

        break



# Muestra solo números impares - 1,3,5,7,9

for x in range(10):

    # Check if x is even

    if x % 2 == 0:

        continue

    print(x)