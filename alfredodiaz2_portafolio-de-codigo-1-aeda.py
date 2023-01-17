




#Entrada de datos
dinero = 5560

#Proceso
b500 = dinero // 500; b500i = dinero % 500
b200 = b500i // 200; b200i = b500i % 200
b100 = b200i // 100; b100i = b200i % 100
b50 = b100i // 50; b50i = b100i % 50
b20 = b50i // 20; b20i = b50i % 20


#Salida
if b500 >= 1:
    print(str(b500) + " billete" + ('s' if (b500) > 1 else '') + " de 500")
if b200 >= 1:
    print(str(b200) + " billete" + ('s' if (b200) > 1 else '') + " de 200")
if b100 >= 1:
    print(str(b100) + " billete" + ('s' if (b100) > 1 else '') + " de 100")
if b50 >= 1:
    print(str(b50) + " billete" + ('s' if (b50) > 1 else '') + " de 50")
if b20 >= 1:
    print(str(b20) + " billete" + ('s' if (b20) > 1 else '') + " de 20")

import random as rn
cantidad=300

for i in range(1,cantidad+1):
    ale=rn.randint(500,1000)
    print(ale)

