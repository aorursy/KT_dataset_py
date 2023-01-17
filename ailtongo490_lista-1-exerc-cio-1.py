import numpy as np


def operacoes(lista):

    dados = {}

    dados['maior'] = max(lista)

    dados['soma'] = sum(lista)

    dados['ocorrencias'] = lista.count(lista[0])

    dados['media'] = round(np.mean(lista), 2)

    dados['proximoDaMedia'] = lista[0]

    dados['somaNegativo'] = 0

    dados['vizinhosIguais'] = 0

    dados['sublistas'] = []

    

    distanciaMedia = abs(lista[0] - dados['media'])

    

    for indice,numero in enumerate(lista):

        for i,n in enumerate(lista):

            if indice != i:

                dados['sublistas'].append([numero,n])

            

        if (indice + 1) < len(lista):

            if numero == lista[indice + 1]:

                dados['vizinhosIguais'] += 1

            

        if numero < 0:

            dados['somaNegativo']  += numero

            

        if  abs(numero - dados['media']) < distanciaMedia:

            dados['proximoDaMedia'] = numero

            distanciaMedia = abs(numero - dados['media'])

    

    

    return dados
operacoes([1,2,2,3,5,4,1,1,-50,4,6,4,1,2.95,-10])
[1,4,6,7,4].count(4)
for pos,i in enumerate([1,2,3,5,4,1,1,-50,4,6,4,1,2.95,-10]):

    print("%d - %d" % (pos,i))
a = [1,2,3,5,4,1,1,-50,4,6,4,1,2.95,-10]

len(a)