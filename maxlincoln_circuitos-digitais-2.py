import pandas as pd

from random import randint
dataset = pd.read_csv('../input/training.csv')

dataset['plaintext_id'].head()

palavras = {}

'''44680'''

for i in range(0, 100):

    palavras[i] = dataset['plaintext_id'][i]

for i in range(0, 100):

    if(i%10==0):

        print("")

        print("Bloco " + str(i//10))

    print(str(i)+" : "+palavras[i])
print(palavras)
class Bloco: 

    def __init__(self,cod):

        self.cod = cod

        self.palavras = []

blocos=[]

palavrasPorBloco=[]

for i in range(0,100):

    z = i * 10

    blocos.append(Bloco(i))

    for k in range(z,z+10):

        blocos[i].palavras.append(palavras[k])

    

print("Exemplo de Bloco:")

print(blocos[0].palavras)



    

cache = []

    

b1 = randint(0,4467)

b2 = randint(0,4467)

b3 = randint(0,4467)

b4 = randint(0,4467)



cache.append(blocos[b1])

cache.append(blocos[b2])

cache.append(blocos[b3])

cache.append(blocos[b4])



print("Blocos iniciados na cache: [" +str(cache[0].cod)+", "+str(cache[1].cod)+", "+str(cache[2].cod)+", "+str(cache[3].cod)+"]")

print("")

print("Plavraas do bloco ["+str(cache[0].cod)+"] :")

print(cache[0].palavras)

print("")

print("Plavraas do bloco ["+str(cache[1].cod)+"] :")

print(cache[1].palavras)

print("")

print("Plavraas do bloco ["+str(cache[2].cod)+"] :")

print(cache[2].palavras)

print("")

print("Plavraas do bloco ["+str(cache[3].cod)+"] :")

print(cache[3].palavras)
def buscarNaMemoriaPrincipal(entrada):

    if(entrada in palavras):

        x = entrada // 10

        print("")

        print("")

        print("Bloco da palavra: " + str(x))

        print("")

        print("Palavra requisitada pela CPU: " + palavras[entrada] + " (Encontrada na memorória principal)")

        print("")

        print("Demais Plavras do bloco: ")

        print(blocos[x].palavras)

        cache[3]=cache[2]

        cache[2]=cache[1]

        cache[1]=cache[0]

        cache[0]=blocos[x]

        cache

        print("")

        print("Blocos na cache: [" +str(cache[0].cod)+", "+str(cache[1].cod)+", "+str(cache[2].cod)+", "+str(cache[3].cod)+"]")

    else:

        print("Palavra não encontrada na memória principal")

        

def buscarNaCache(entrada):

    bloco = entrada // 10

    cont = 0

    for i in range(0, 4):

        if(bloco == cache[i].cod):

            cont = cont + 1

            print("")

            print("")

            print("Bloco da palavra: " + str(bloco))

            print("")

            print("Palavra requisitada pela CPU: " + palavras[entrada] +" (Encontrada na cache)")

            print("")

            print("Demais Plavras do bloco: ")

            print(cache[i].palavras)

            print("")

            aux=[]

            aux.append(cache[0])

            aux.append(cache[1])

            aux.append(cache[2])

            aux.append(cache[3])

            cache[0]=cache[i]

            del(aux[i])

            for k in range (1, 4):

                cache[k]=aux[k-1]

                

            print("Blocos na cache: [" +str(cache[0].cod)+", "+str(cache[1].cod)+", "+str(cache[2].cod)+", "+str(cache[3].cod)+"]")  

            break

    if(cont == 0):

        buscarNaMemoriaPrincipal(entrada)
def validar(entrada):

    try:

        int(entrada)

    except ValueError:

        print("Endereço inválido")

        return False

    return True
entrada = input()

if(validar(entrada)):

    entrada=int(entrada)

    buscarNaCache(entrada)




