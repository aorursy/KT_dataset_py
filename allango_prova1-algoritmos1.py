# Faça o código da primeira questão aqui

print("Entre com um número inteiro positivo que representa os segundos a serem convertidos em horas,minutos e segundos")

n = int(input())



h = n // 3600;   # parte inteira da divisao



resto = (n % 3600)     # resto da divisao



m = resto // 60



s = (resto % 60)



print(n," segundos representa ",h," horas, ", m, " minutos e ", s, " segundos")

            
# Faça o código da segunda questão aqui



for x in range(1,11):

    hx = (x**2) - 16

    if (hx < 0):

        fx = 1

    else:

        fx = hx

    if(fx > 0):

        gx = 0

    elif (fx == 0):

        gx = (x*x) + 16

    y = fx + gx

    print ("x=",x," y=",y)

      

    
# Faça o código da terceira questão aqui

dec = int(input())

bin = 0

fator = 1

while (dec > 0): 

    while (dec>0):

        bin = bin + (dec % 2)*fator

        dec = dec // 2

        fator = fator*10

    print(bin)    

    dec = int(input())

    bin = 0

    fator = 1
import math

# Faça o código da quarta questão aqui



A = 0.1

while( A <= 6.3):

    angulo = A * (math.pi/180)

    #print(angulo)

    i = 3

    topo = angulo**(i)

    n = 3

    fat = n - 1

    cont = fat - 1

    sinal = -1

    while cont > 0 :

      fat = fat * cont

      cont = cont - 1

    fatorial = fat * n 

    parcial= topo / fatorial

    a0 = 0

    a1 = angulo - parcial

    erro = abs (a1 - a0)

    parada = 10**(-35)

    while erro>parada:

        sinal = sinal * (-1)

        a0 = a1 

        i = i + 2

        n = n + 2

        fat = n - 1

        cont = fat - 1

        while cont > 0 :

            fat = fat * cont

            cont = cont - 1

            fatorial = fat * n 

        topo = angulo**(i)

        parcial=  (topo/fatorial)

        a1 = a0 + parcial

        erro = abs (a1 - a0)

    print ("Angulo:", A,"seno:", a1)

    A = A + 0.1