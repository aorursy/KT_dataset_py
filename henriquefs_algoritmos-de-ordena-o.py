from random import randint

n = 5

vet = [0] * n

for i in range(n):

    vet[i] = randint(1,10)

print(vet)    

for j in range(n-1):

    for k in range(j+1,n):

        if vet[j] > vet[k]:

            aux = vet[j]

            vet[j] = vet[k]

            vet[k] = aux

print(vet)
#Bubble Sort

#Um "for" e um "while"



from random import randint

n = 5

vet = [0] * n

troca = 1

for i in range(n):

    vet[i] = randint(1,10)

print(vet)    

while troca == 1:

    troca = 0

    for j in range(n-1):

        if vet[j] > vet[j+1]:

            aux = vet[j]

            vet[j] = vet[j+1]

            vet[j+1] = aux

            troca = 1

print(vet)