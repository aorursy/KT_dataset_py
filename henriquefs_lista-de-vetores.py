#Ex 01



n = int(input())

vet = [0] * n

soma = 0

acima = 0

abaixo = 0



for i in range(n):

    vet[i] = float(input())

    soma += vet[i]



media = soma / n



for j in range(n):

    if vet[j] >= media:

        acima += 1

    else:

        abaixo += 1

        

print('Acima ou igual:',acima)

print('Abaixo:',abaixo)
# Ex 02



n = int(input())

vet = [0] * n



for i in range(n):

    vet[i] = float(input())

    

for j in range(n-1,-1,-1):

    print(vet[j])
#Ex 03



vet = [0] * 500

cont = -1

ent = int(input())



while cont < 500 and ent > 0:

    cont += 1

    vet[cont] = ent

    ent = float(input())

    

for i in range(cont,-1,-1):

    print(vet[i])
#Ex 05



n = 4

vet = [0] * n

vet2 = [0] * n

perdeu = False

frase = 'Vitória!!!'

for i in range(n):

    vet[i] = float(input())

for j in range(n):

    vet2[j] = float(input())

    if vet[j] != vet2[j]:

        perdeu = True

        frase = 'Derrota :('

    

print(frase)
#Ex 06



n = 5

vet = [0] * n

vet2 = [0] * n

perdeu = False

msg = 'Vitória!!!'



for i in range(n):

    vet[i] = int(input())

for j in range(n):

    vet2[j] = int(input())

    

for k in range(n-1):

    for l in range(k+1,n):

        if vet[l] < vet[k]:

            aux = vet[k]

            vet[k] = vet[l]

            vet[l] = aux

        if vet2[l] < vet2[k]:

            aux = vet2[k]

            vet2[k] = vet2[l]

            vet2[l] = aux



m = 0

while m < n and not perdeu:

    if vet[m] != vet2[m]:

        perdeu = True

        msg = 'Derrota :('

    m += 1

            

print(vet)

print(vet2)

print(msg)
# Ex 07



n = 5

vet = [0] * n

for i in range(n):

    vet[i] = int(input())

for j in range(n-1):

    for k in range(j+1,n):

        if vet[k] < vet[j]:

            aux = vet[k]

            vet[k] = vet[j]

            vet[j] = aux

indiceMediana = n // 2

print(vet)

print(vet[indiceMediana])
#Ex 08

n = int(input())

while n < 1 or n > 30:

    print('Número Inválido, repita:')

    n = int(input())

vet = [0] * n



for i in range(n):

    vet[i] = int(input())

    j = 0

    repetido = False

    while j < i and not repetido:

        if vet[i] == vet[j]:

            repetido = True

            print('Valor repetido, digite outro:')

        j += 1

    while repetido:

        vet[i] = int(input())

        j = 0

        repetido = False

        while j < i and not repetido:

            if vet[i] == vet[j]:

                repetido = True

                print('Valor repetido, digite outro:')

            j += 1
# Ex 09



from random import randint



cont = 0

n = int(input())

while n < 0 or n > 380:

    n = int(input())

vet = [None] * n

modas = [None] * n



for i in range(n):

    vet[i] = int(input())



qntdDeModa = 0

for j in range(n):

    aux = 0

    if vet[j] != None:

        atual = vet[j]

        for k in range(n):

            if atual == vet[k]:

                aux += 1

                vet[k] = None

        if aux == qntdDeModa:

            cont += 1

            modas[cont] = atual

        if aux > qntdDeModa:

            for m in range(cont+1):

                modas[m] = None

            cont = 0

            qntdDeModa = aux

            modas[0] = atual

                    

print('Modas:',modas)

print('Qntd de vezes:',qntdDeModa)
#Ex 10



from random import randint



n = int(input())

while n < 0 or n > 380:

    n = int(input())

vet = [0] * n

for i in range(n):

    vet[i] = float(input())

print(vet)

qntd = 0

for j in range(n):

    aux = 0

    if vet[j] != None:

        atual = vet[j]

        for k in range(n):

            if atual == vet[k]:

                aux += 1

                vet[k] = None

        if aux > qntd:

            qntd = aux

            moda = atual

        if aux == qntd:

            if atual < moda:

                moda = atual

print(moda)
#Ex 11



from random import randint



n = int(input())

while n < 0 or n > 500:

    n = int(input())

for i in range(n):

    vet[i] = randint(1,10)

print(vet)

qntd = 0

for j in range(n):

    aux = 0 

    if vet[j] != None:

        atual = vet[j]

        for k in range(n):        

            if vet[k] == atual:

                aux += 1

                vet[k] = None

        if aux > qntd:

            qntd = aux

            moda = atual

print(moda)

print(qntd)
#Ex 12



from random import randint



soma = 0

n = 5

vet = [None] * n

for i in range(n):

    vet[i] = randint(1,10)

    soma += vet[i]

print(vet)

media = soma / n

dist = abs(media - vet[0])

prox = vet[0]

for j in range(1,n):

    if abs(media - vet[j]) < dist:

        dist = abs(media - vet[j])

        prox = vet[j]

print(media)

print(prox)
#Ex 13



from random import randint



soma = 0

somatorio = 0

n = int(input())

while n < 0 or n > 200:

    n = int(input())

vet = [0] * n

for i in range(n):

    vet[i] = randint(1,10)

    soma += vet[i]

media = soma / n

#Somatória

for j in range(n):

    somatorio += (vet[j] - media) ** 2

desvio = (somatorio / n) ** 1/2

print(vet)

print(media)

print(desvio)
# Ex 14



from random import randint



contido = True

msg = 'Sim'

n = 100

vet = [None] * n

for i in range(n):

    vet[i] = int(input())

    while vet[i] < 0:

        vet[i] = int(input())

j = 1

while j <= 5 and contido:

    aux = 0

    k = 0

    while k < n and contido:

        if j == vet[k]:

            aux += 1

        k += 1

    if aux != 1:

        contidos = False

        msg = 'Não'

    j += 1

print(vet)

print(msg)          
#Ex 15



from random import randint



n = 10

vet = [0] * n

for i in range(n):

    vet[i] = randint(1,10)

print(vet)

for j in range(n-1):

    for k in range(j+1,n):

        if (vet[j] % 2 == 1) and (vet[k] % 2 == 0):

            aux = vet[j]

            vet[j] = vet[k]

            vet[k] = aux

print(vet)
#Ex 16