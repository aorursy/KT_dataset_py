topo = 1

b = 2

result1 = topo / b + 1

result0 = 0

erro = abs (result1 - result0)

K = 10**(-4)

sinal = -1

while erro > K :

    b = b +2

    result0 = result1

    result1 = (topo/b) * sinal

    result1 = result1 + result0

    erro = abs (result1 - result0)

    sinal = sinal * (-1)

    print (result1)
topo = 1

b = 2

result1 = topo / b * (-1)

result1 = result1 + 1

result0 = 0

sinal = -1

b = 2

while b <= 200:

    result0 = result1

    b= b + 2

    topo = 1

    sinal = sinal * (-1) 



    result1 = (topo/b) * sinal

    result1 = result1 + result0

print (result1)
sinal = -1

topo = 1 

base =2

total = (topo /base) *(-1)

total = total +1

guarda = 0

while base <=200:

    guarda = total

    base = base + 2

    sinal = sinal *(-1)

    total = (topo / base) * sinal

    total = total + guarda

    print(total)
n = int ( input ( 'Qual número você saber? ') )

k = 1 

if n > 1:

  j = 1

  print (1)

if n > 2:

  j = 1

  print (1)

a = 2

b = 1

c = a + b

if n > 3:

  print (3)

while k < n - 2:

  j = a + b

  k = k + 1

  b = a 

  a = j

  print (j)

n = int (input ( ' Qual fatorial calcular '))



x = 1

fat = n - 1

cont = fat - 1

while cont > 0 :

  fat = fat * cont

  cont = cont - 1

fatorial = fat * n 

  

print (fatorial)
a = float ( input ( ' Qual a raiz ? ' ))

x0 = 1

b = x0 + (a/x0)

xi = 0.5 * b

erro = abs (x0- xi)

parada = 10**(-9)

while erro > parada :

  x0 = xi 

  b = x0 + (a/x0)

  xi = 0.5 * b

  erro = abs (x0- xi)

print (xi)



c = float (input ('Digite um valor Entre 0 e 2! '))

a0 = 1

c0 = 1 - c

c1 = c0 ** 2

b = 1 + c0

a1 = a0 * b

erro = abs (a0 - a1)

parada = 10**(-9)

while erro > parada :

  a0 = a1 

  c0 = c1

  b = 1 + c1

  a1 = a0 * b

  c1 = c1 ** 2

  erro = abs (a1 - a0)

print (a1)

  
s = 1 

topo = 1

base = 1

result1 = topo / base

result0 = 0

while topo <99:

  topo = topo + 2 

  base = base + 1

  result0 = result1

  result1 = topo/base + result0

print (result1)
result0 = 0

base = 50

topo = 1

ok = 2

result1 = ok**(topo)/base

while topo <50:

  result0 = result1

  topo = topo + 1

  base = base - 1

  result1 = ok**(topo)/base + result0

  print(result1)

print (result1)
result0 = 0

topo1 = 37

topo2 = 38

base = 1

result1 = (topo1 * topo2) / base

while topo1 > 1:

  result0 = result1

  topo1 = topo1 - 1

  topo2 = topo2 - 1

  base = base + 1

  result1 = ((topo1 * topo2) / base) + result0

print (result1)

result0 = 0

topo = 1

base = 1 

aum = 3

i = 1

result1 = topo / base

while base <100:

  i = i + 1

  result0 = result1

  if topo > 0:

    topo = 1 + topo

  if topo <0:

    topo = topo - 1 

  topo = topo * -1

  base = base + aum

  aum = aum + 2

  result1 = topo / base +  result0

print (result1)



result0 = 0

x = 1

topo = 1000

base = 1

result1 = 1000

x = 1

print (result1)

while x < 50:

  x = x + 1

  result0 = result1

  topo = topo -3

  if base >= 1:

    base = base + 1

  if base <= 1:

    base = base - 1

  base = base * -1

  result1 = (topo/base) + result0

  print (result1)

result0 = 0

topo = 480

base = 10

x = 1

result1 = topo / base

while 30>x:

  x = x + 1

  result0 = result1 

  if topo >= 0:

    topo = topo - 5

  if topo <0:

    topo = topo + 5

  topo = topo * -1

  base = base + 1 

  result1 = topo / base + result0

  print (result1)

result0 = 0

topo = -4

base = 3

result1 = 4 + (topo / base)

parada = 10**(-4)

erro = abs (result0 - result1)



while erro > parada:

  result0 = result1

  topo = topo * -1

  base = base + 2 

  result1 = topo / base + result0

  erro = abs (result0 - result1)

print (result1)

  

  
result0 = 0

topo = 1

base = 1

result1 = topo / base

x = 1

while x  < 15:

  x = 1 + x 

  result0 = result1

  topo = topo * -1

  base = base + 2 

  base = base **3

  result1 = topo / base + result0

S =  result1 * 32

s = S**(1/3)

print (s)

  

  
result0 = 0

topo = float (input ('Digite um valor ' ))

base = 1

i = 25

topo = topo **(i)

result1 = topo / base    

while base <25:

  result0 = result1

  base = 1 + base

  i = i - 1

  if i % 2 == 0:

    topo= topo **(i)

    topo = topo * -1

  topo = topo ** (i)

  result1 = topo / base + result0

  

print (result1)



result0 = 0

to  

po = 1

base = 15

result1 = topo / (base**2)



while topo < 16384:

  result0 = result1

  if base > 0:

    base = base - 1

  if base < 0:

    base = base + 1 

  base = base * -1

  b = base **2

  topo = topo * 2

  result1 = result0 + topo /b

  print (result1)



print (result1)
