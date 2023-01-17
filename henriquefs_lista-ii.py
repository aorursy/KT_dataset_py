#Ex 01



k = 1

n = 1

c0 = 2

t0 = 1

c1 = c0 + 2

t1 = 1/c0 * -1 ** (n - 1)

soma = t0

E = abs(t1 - t0)





while E <= k:

    n = n + 1

    c0 = c1

    t0 = t1

    c1 = c0 + 2

    t1 = 1/c0 * -1 ** (n - 1)

    soma = soma + t0

    E = abs(t1 - t0)

    

print(soma)
#Ex 02



d = 0

soma = 1

control = -1



while d < 200:

    d = d + 2

    num = 1 / d * control

    control = control * -1

    soma = soma + num

    

print(soma)
#Ex 03



n = 15

cont = 4

f1 = 1

f2 = 1

f3 = f1 + f2



if n > 2:

    print(f1)

    print(f2)

    print(f3)

    while cont <= n:

        f1 = f2

        f2 = f3

        f3 = f1 + f2

        print(f3)

        cont = cont + 1

elif n == 2:

    print(f1)

    print(f2)

else:

    print(f1)
#Ex 04



n = int(input('Informe um número: '))

f0 = 1

f1 = f0

cont = 1



while cont <= n:

    f0 = f1

    f1 = f0 * cont

    cont = cont + 1

    

print('Fatorial de {}: {}'.format(n, f1))
#Ex 05



a = float(input('Informe um número: '))

x0 = 1

x1 = 1/2 * (x0 + a/x0)

e = abs(x1 - x0)



while e >= 10 ** -6:

    x0 = x1

    x1 = 1/2 * (x0 + a/x0)

    e = abs(x1 - x0)

    

print('Raiz quadrada de {}: {}'.format(a, x0))
#Ex 06



x = float(input('Informe um número entre 0 e 2: '))

c0 = 1 - x

a0 = 1

c1 = c0 ** 2

a1 = a0 * (1 + c0)

E = abs(a1 - a0)



while E >= 10 ** -9:

    c0 = c1

    a0 = a1

    c1 = c0 ** 2

    a1 = a0 * (1 + c0)

    E = abs(a1 - a0)

    

print('Inverso de {}: {}'.format(x, a0))



#Ex 07



n1 = 1

d1 = 1

soma = n1/d1



while n1 < 99:

    n0 = n1

    d0 = d1

    n1 = n0 + 2

    d1 = n0 + 1

    soma = soma + n1/d1

    

print(soma)
#Ex 08



e1 = 1 

d1 = 50

n1 =  2 ** e1 / d1

soma = n1



while d1 > 1:

    e0 = e1

    d0 = d1

    e1 = e0 + 1

    d1 = d0 - 1

    n1 = 2 ** e1 / d1

    soma = soma + n1

    

print(soma)

    
#Ex 09



n0 = 38

n1 = 37

d1 = 1

num1 = n0 * n1 / d1

soma = num1



while n1 > 1:

    n0 = n1

    n1 = n0 - 1

    d0 = d1

    d1 = d0 + 1

    num1 = n0 * n1 / d1

    soma = soma + num1

    

print(soma)



    
#Ex 10



cont = 1

num = 1

soma = num



while cont < 10:

    cont = cont + 1

    num = (cont / cont**2) * (-1**(cont+1))

    soma = soma + num

    

print(soma)

    
#Ex 11



cont = 1

n1 = 1000

d1 = 1

num = n1 / d1

soma = num



while cont < 50:

    cont = cont + 1

    n0 = n1

    d0 = d1

    num = n1 / d1 * (-1 ** cont-1)

    soma = soma + num

    

print(soma)
#Ex 12



cont = 1

n1 = 480

d1 = 10

num = n1 / d1

soma = num



while cont < 30:

    cont = cont + 1

    n0 = n1

    d0 = d1

    n1 = n0 - 5

    d1 = d0 + 1

    num = n1 / d1 * -1 ** cont-1

    soma = soma + num

    

print(soma)

    

#Ex 13



cont = 1

d1 = 1

num = 4 / d1 * -1 ** (cont - 1)

mod = abs(num)

soma = 0



while mod >= 0.0001:

    soma = soma + num

    cont = cont + 1

    d0 = d1

    d1 = d0 + 2

    num = 4 / d1 * -1 ** (cont - 1)

    mod = abs(num)

    

print(soma)
#Ex 14



cont = 1

d1 = 1

num = 1

soma = num



while cont < 51:

    cont = cont + 1

    d0 = d1

    d1 = d0 + 2

    num = 1 / d1 ** 3 * -1 ** (cont-1)

    soma = soma + num

    

pi = (soma * 32) ** (1/3)



print(pi)

#Ex 15



x = float(input('Informe um número: '))

cont = 1

num = x ** 25 / cont

soma = num



while cont < 25:

    cont = cont + 1

    num = x ** (26 - cont) / cont * (-1) ** (cont - 1)

    soma = soma + num

    

print(soma)

    
#Ex 16



cont = 1

num = cont / (16 - cont) ** 2

soma = num



while cont < 15:

    cont = cont + 1

    num = cont / (16 - cont) ** 2 * -1 ** (cont-1)

    soma = soma + num

    

print(soma)