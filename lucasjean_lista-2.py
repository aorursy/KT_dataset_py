s = 1

n -= 2 

x = 0

while(n <= 200):

 x =  (1 / n)

 s = s + x

 n = n + 2

 n = n * -1

 

print(s)    
n = int(input())

f1 = 1

f2 = 1

f0 = 0

fn = 0

cont = 3

if(n <= 2):

    print(fn)

    print(f2)

    print(f0)

else:

  while(cont <= n):

    fn = f1 + f2 

    f1 = f2

    f2 = fn

    f0 = f2 - f1

    cont += 1

print(fn)

print(f1)

print(f0)
x = int(input())

cont = 1

while (x >= 1):

    n= x-1    

    fat = x * n

    x = n -1

    cont = cont * fat

print(cont)
k = float(input())

x = -1

y = 2

s = 1

e = 1

while e <= k:

  e = k + (x/(y+2))

  k = x/y * -1 

  s += (x/y) * -1

  y +=2 

    

    

print(s)
s = 1

x = 50

y = 1

cont = 0

while x >= 1:

 s = (2**y)/x 

 x -= 2

 y += 2

 cont += s

print(cont)
soma = 0