# Faça o código da primeira questão aqui

n = int (input (' Digite um número inteiro '))

horas = 0

minutos = 60 

segundos = n 

minutos = segundos // 60

horas = minutos // 60

if horas > 1:

    n = n - horas * 3600

    segundos = n 

    minutos = segundos // 60

if minutos > 1:

    n = n - (minutos *60)

segundos = n 

if minutos ==60:

    minutos = 0

if minutos > 60:

    minutos = minutos - 60

if segundos ==60:

    segundos = 0

if segundos >60:

    segundos = segundos - 60

    

print ('horas', horas ,'minutos', minutos ,'segundos', segundos)
# Faça o código da segunda questão aqui

x = 1

h0 = 0

h1 = x**(2)

h1 = h1 - 16

f0 = 0

f1 = 0

if h1 >= 0:

    f1 = h1

if h1 <0:

    f1 = 1

g0 = 0

g1 = 0

if f1 ==0:

    g1 = x **(2) + 16

if f1 >0:

    g1 = 0

h0 = h1

f0 = f1

g0 = g1

y0 = 0

y1 = g1 + f1

while x <10:

    x = x + 1 

    y0 = y1

    h0 = h1

    f0 = f1

    g0 = g1

    h1 = x**(2)- 16 

    if h1 >= 0:

        f1 = h1

    if h1 <0:

        f1 = 1

    if f1 ==0:

        g1 = 32

    if f1 >0:

        g1 = 0

    if f1<0:

        g1 = 0

    y1 = g1 + f1 + y0

    print(y1)

    

print(y1)



   

# Faça o código da terceira questão aqui

n = int (input ('Digite um valor ' ))

k = str(n)

j = k[::-1]

binario = n % 2

k0 = str (0)

k1 = str (binario) 

bino  = n//2

while bino>= 2:

    k0 = k1

    binario = bino % 2

    bino = bino //2

    k1 = str(binario) + k0

if bino ==1:

    k0 = k1

    k1 = '1' + k0

print (k1)

while n > 0:

    n = int (input ('Digite um valor ' ))

    k = str(n)

    j = k[::-1]

    binario = n % 2

    k0 = str (0)

    k1 = str (binario) 

    bino  = n//2

    while bino>= 2:

        k0 = k1

        binario = bino % 2

        bino = bino //2

        k1 = str(binario) + k0

    if bino ==1:

        k0 = k1

        k1 = '1' + k0

    print (k1 , n)
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
