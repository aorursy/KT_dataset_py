n = int(input())

while (n< 1 or n > 100):

    print("Digite um valor entre 1 e 100")

    n = int(input())

v = [None]*n

soma = 0

for i in range(n):

    v[i] = int(input())

    soma = soma + v[i]

media = soma/n

acima = 0

abaixo = 0

for i in range(n):

    if  v[i] >= media:

        print(v[i], "está acima da média que é ", media)

        acima = acima + 1

    else:

        abaixo = abaixo + 1

        print(v[i], "está abaixo da média que é", media)

print(acima," estão acima da média")

print(abaixo," estão abaixo da média")

n = int(input())

while (n< 1 or n > 100):

    print("Digite um valor entre 1 e 100")

    n = int(input())

v = [None]*n

soma = 0

for i in range(n):

    v[i] = int(input())

for i in range(n-1,-1,-1):

    print(v[i])



    
n = int(input())

x = [None]*n

soma = 0

for i in range(n):

    x[i] = int(input())

    soma = soma + x[i]

media = soma/n

soma = 0

for i in range(n):

    soma = soma + x[i]

soma = soma/n

desvio = soma**(0.5)

print("O valor do desvio padrão é:",desvio)
n = int(input("Digite um valor entre 1 e 750"))

while(n < 1 or n > 750):

    n = int(input("Digite um valor entre 1 e 750"))

v = [None]*n

for i in range(n):

    v[i] = int(input())

moda = v[0]

qtde = 0

for i in range(n):

    aux = 0

    for j in range(n):

         if (v[j]!= None and v[i] == v[j]):

            aux = aux + 1

            moda = v[i]

    if (aux >= qtde):

        moda = v[i]

        qtde = aux

        v[i] = None

        print("Moda ", moda, "Quantidade: ", qtde)
i = None



if (i == None):

    print("sim")



v = [None]*500

i = 0

valor = float(input())

while (i < 500 and valor >= 0):

    v[i] = valor

    i = i+1

    valor = float(input())



for j in range(i, -1, -1):

    print(v[j])

 