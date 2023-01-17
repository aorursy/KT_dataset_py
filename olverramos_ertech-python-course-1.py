round(45/7,2)
# Respuesta aqui:

213/6
#n= (a+b+c+d+e+f)/6 
(41+40+40+39+25+23)/6

#Mi Nombre

'''

aaaaaa

'''

print("""Camilo\'s Peluquería

Tiene su inauguración el día de mañana

Traer sus tanguitas narizonas

Camilo va a evaluar si les queda bien

o mal...

    

Nos vemos!!!!

""")
nombre = 'Cristiano'

apellido = 'Messi'

edad = 35

datos = "El Señor " + nombre + ' ' + apellido + ' tiene una edad de ' + str(edad) + " años"

datos2 = f"El Señor {nombre} {apellido} tiene una edad de {edad} años"



datos = "El Señor "



#datos = datos + nombre

datos += nombre



print (datos2)

print (datos)



a = 6

a += 5

a -= 3

a *= 2

print (a)
a = [1,3,"Perro",1,6,3.7,1,2,4]

print (a)

a[3:8]

a.append(4)

print (a)

a.insert(3, 67 )

print (a)

del a[3]

print (a)
t = (1,6,3.7,1,2,4)

t.append(6)
(a, b) = (5, 6)

print (a)

print (b)
x = 40

(x != 30) or not (x < 30)
x = 40

if (x != 30) and not (x < 30):

    print("Hola")

    if x >= 40:

        print("Mundo")

elif x < 15:

    print ("Pruebe otra cosa 15")

elif x < 25:

    print ("Pruebe otra cosa 25")

else:

    print("No Funciona")
i = 0

while i < 10:

    print (i)

    i += 1
n = 1000

i = 0

result = 0

while i < n:

    result += i

    i += 1

print(result)
# R/ Coloque el código aqui



#n = 45

#n = 55

#n = 4657221

#n = 438323842397427347239842793842936

#n = 1379217

n = 57

resultado = True

i = 2

while i <= n / 2:

    if (n % i) == 0:

        print (f"{n} es divisible por {i}")

        resultado = False

    i += 1

if resultado:

    print (f"{n} es un número primo")

else:

    print (f"{n} no es un número primo")

#R/ Coloque el código aqui

m = 10

j = 0

k = 2



while j < m:

    es_primo = True

    i = 2

    while i <= k / 2:

        if (k % i) == 0:

            es_primo = False

        i += 1

    

    if es_primo == True:

        print (k)

        j += 1

    k += 1

1 % 2