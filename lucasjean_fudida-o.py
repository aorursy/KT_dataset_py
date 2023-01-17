a = 38

b = 39

c = 1

calculo = 0

s = 0

while a >= 2:

  calculo = (a * b) / c

  a -= 1

  b -= 1

  c += 1

  s += calculo 

print(s)
a = 480

b = 10

calculo = 0

cont = 1

conta = 0

mult = -1

while cont <= 30:

    calculo = a / b

    conta += calculo

    a -= 5

    b += 1

    cont += 1

    print(calculo)

print(conta)    
s = 1 

y = 5

x = 3

cont = 1

while cont <= 50:

    calculo = (1 / (x**3)) + (1/(y**3))

    cont += 1

    x += 4

    y += 4

    s -= calculo

print(s)

H = (s*32)**(1/3)

print(H)