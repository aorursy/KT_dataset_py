###Programa que imprima los 25 primeros numeros naturales

n = 1

while n <= 25: 

    print (n),

    n += 1
###Imprimir los numeros impares desde el 1 al 25, ambos inclusive

n = 1

h = ''

while n <= 25:

    if n%2 != 0:

        h += ' %i' % n

    n += 1

print (h)
###Imprimir los numeros pares desde el 40 hasta el 60, ambos inclusive

n = 40

h = ''

while n <= 60:

    if n%2 == 0:

        h += ' %i' % n

    n += 1

print (h)
###Imprimir los numeros 48, 52, 56, ..., 120

n = 48

h = ''

while n <= 120:

    h += ' %i' % n

    n += 4

print (h)

 
###Imprimir los numeros 100, 95, 90, ..., 20

n = 100

h = ''

while n >= 20:

    h += ' %i' % n

    n -= 5

print (h)
###Calcular e imprimir la suma 1+2+3+4+5+...+50

h = range(1, 51)

print (sum (h)) 

###Calcular e imprimir el producto 1*2*3*4*5*...*20

n = 1

h = 1

while n <= 20:

    h *= n

    n += 1

print (h)

 
###Calcular e imprimir la suma 50+48+46+44+...+20

n = 50

h = 0

while n >= 20:

    h += n

    n -= 2

print (h)
###Programa que imprima los nuumeros impares desde el 100 hasta la unidad y calcule su suma

n = 100

h = 0

while n >= 1:

    if n%2 != 0:

        print (n),

        h += n

    n -= 1

print ('y su suma es: %i' % h)
##Imprimir los numeros del 1 al 100 y calcular la suma de todos los nuumeros 

###pares por un lado, y por otro, la de los impares.

n = 1

p = 0

i = 0

while n <= 100:

    print (n),

    if n%2 == 0:

        p += n

    else:

        i += n

    n += 1

print ('La suma de los pares es igual a %i' % p)

print ('La suma de los impares es igual a %i' % i)