numero=78

def es_primo(num):

    contador=0

    for i in range(1,num+1):

        if num % 1== 0:

            contador += 1

    if contador==2:

        print("El numero es primo")

    else:

        print("El numero no es primo")

print(es_primo(78))
serie=(5,3,7,2,6,3)

print("El numero mayor es",max(serie))

print("El numero menor es",min(serie))

sorted([5,3,7,2,6,3])



list=(5,4,7,2,8,4,6)



media=((sum(list))/7)

print("La media es",media,)



    



list=['Juan','Pedro','Ana','Alejandra','Raul']

sorted(list)

dinero = int(input("Precio Electrodomestico: "))

list = [500, 200, 100, 50, 20]



for i in list:

    billetes = dinero * i

    dinero = dinero % i

print("Usted pagara con",str(billetes),"billetes de" ,str(i))
dinero = int(input("Precio Electrodomestico: "))

list = [200,50, 20]



for i in list:

    billetes = dinero * i

    dinero = dinero % i

print("Usted pagara con",str(billetes),"billetes de" ,str(i))
import random as rn

suma=0

cantidad=300



for i in range(1,cantidad+1):

    ale=rn.randint(500,1000)

    import numpy as np

    list=[ale]

    arreglo=np.array(list)

    print(arreglo)



print("El numero mayor es",max(arreglo))

print("El numero menor es",min(arreglo))