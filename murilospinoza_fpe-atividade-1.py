import numpy as np

import pandas as pd
def encontraModa(Vpd):

    contagem = Vpd.value_counts()

    valores_repeticoes = dict(contagem[contagem == max(contagem)])



    conector = ''

    moda = ''

    

    # Converte {38: 2, 49: 2} em "38 (x2), 49 (x2)"

    for valor in valores_repeticoes:

        moda += conector + "{:8.4f}".format(valor) + ' (x' + str(valores_repeticoes[valor]) +')'

        conector = ', '



    return moda
def analisa(valores):   

    Vpd = pd.Series(valores)



    analise = { }



    analise['media'] = np.average(valores)

    analise['moda'] = encontraModa(Vpd)

    analise['mediana'] = Vpd.median()



    analise['desvio_padrao'] = np.std(valores)

    analise['variancia'] = analise['desvio_padrao'] ** 2

    analise['coeficiente_variacao'] = analise['desvio_padrao']/analise['media']



    return analise
def imprimeAnalise(numeros, analise):

    valores = [str(i) for i in numeros]

    print('\n\n' + ', '.join(valores) + '\n')

    

    def formataValor(valor):

        return "{:8.4f}".format(valor)



    print(' Média......................: ' + formataValor(analise['media']) )

    print(' Moda.......................: ' + analise['moda'])

    print(' Mediana....................: ' + formataValor(analise['mediana']) )

    print(' Variância amostral.........: ' + formataValor(analise['variancia']) )

    print(' Desvio padrão amostral.....: ' + formataValor(analise['desvio_padrao']) )

    print(' Coeficiente de variação....: ' + formataValor(analise['coeficiente_variacao']) )
VALIDA_NUMEROS_ERROS = {

    1: 'Nenhum número se repetiu',

    2: 'Há menos de 6 números únicos',

    3: 'Números selecionados por outro aluno'

}



def validaNumeros(numeros, turma):

    qtde_unicos = 0



    for numero in numeros:

        if numeros.count(numero) == 1:

            qtde_unicos += 1



    status = 0

    if qtde_unicos == len(numeros):

        status = 1

    elif qtde_unicos < 6:

        status = 2

    elif turma.count(numeros) > 1:

        status = 3



    return status
def analisaTurma(turma):

    for numeros in turma:

        validacao = validaNumeros(numeros, turma)

        if validacao == 0:

            analise = analisa(numeros)

            imprimeAnalise(numeros, analise)

        else:

            print('\n\n' + ', '.join([str(i) for i in numeros]) + '\n')

            print('Conjunto inválido: ' + VALIDA_NUMEROS_ERROS[validacao])
def buscaTurmaEmArquivo(arquivo):

    arquivo_ref = open(arquivo,'r')

    linha = arquivo_ref.readline()

    turma = []



    while linha:

        numeros = linha.split()

        turma.append( [int(i) for i in numeros] )

        linha = arquivo_ref.readline()

    

    return turma
meus_numeros = [8, 9, 10, 14, 16, 20, 20, 27, 27, 33, 46, 48]

analise = analisa(meus_numeros)

imprimeAnalise(meus_numeros, analise)
arquivo = '../input/fundamentos-programacao-estatistica/FPE-atividade-1.txt'

turma = buscaTurmaEmArquivo(arquivo)

analisaTurma(turma)
def geraNumeroUnico(numeros, num_min, num_max):

    num_max += 1

    numero = np.random.randint(num_min, num_max)

    

    # Enquanto o número já estiver na lista, gera um novo

    while numero in numeros:

        numero = np.random.randint(num_min, num_max)



    return numero
def geraNumerosComRepeticoes(comprimento, qtde_repeticoes, num_min, num_max):

    numeros = []



    qtde_unicos = comprimento - qtde_repeticoes

    

    for x in range(qtde_unicos):

        numero = geraNumeroUnico(numeros, num_min, num_max)

        numeros.append(numero)



    for x in range(qtde_repeticoes):

        # Sorteia um número para ser repetido

        i = np.random.randint(0, len(numeros) - 1)

        numeros.append(numeros[i])



    numeros.sort()

    return numeros
def geraTurma():

    turma = []

    turma_tamanho = 50

    numeros_por_aluno, qtde_numeros_unicos = 12, 6

    min_repeticoes, limite_repeticoes = 1, 3

    num_min, num_max = 1, 50



    for x in range(turma_tamanho):

        qtde_repeticoes = np.random.randint(min_repeticoes, limite_repeticoes+1)



        numeros = geraNumerosComRepeticoes(numeros_por_aluno, qtde_repeticoes, num_min, num_max)



        while numeros in turma:

            numeros = geraNumerosComRepeticoes(numeros_por_aluno, qtde_repeticoes, num_min, num_max)



        turma.append(numeros)



    turma.sort()

    return turma
turma = geraTurma()

analisaTurma(turma)