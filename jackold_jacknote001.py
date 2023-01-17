# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 1º desafio...

print ("Bom Dia !")
# 2º desafio

print ('Buenos dias...')
# 3º desafio...

cidade = 'Brasília'

UF = 'DF'

print ('Moro em ',cidade,'/',UF)

print (cidade[0:6])
# 4º desafio...

moeda = 'Dolar'

cotacao = 3.84

print (type(moeda))

print (type(cotacao))

resultado = cotacao * 3

print (type(resultado))

print ('Resultado', resultado)
# 5º desafio...

moeda = 'Dolar'

cotacao = 3.84

cotacao = int(cotacao)

resultado = cotacao * 3

print (cotacao)

print ('Resultado', resultado)
# 6º desafio...

valor = 2+2+10%3*4

print (valor)
# 7º desafio...

def menor_diferenca(a, b, c):

    """Retorna a menor diferença absoluta entre 2 números entre a, b e c."""

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)

print (menor_diferenca(11,3,6))

print (menor_diferenca(6,3,12))

help(menor_diferenca)
# 8º desafio...

def valor_2casas(valor = 3.14159265358979323846264338327950288):

    """Retorna o valor arredondado em 2 casas decimais."""

    return round(valor, 2)

print (valor_2casas())

print (valor_2casas(5.13145))
# 9º desafio...

# Qual o erro desta linha abaixo?

ruound(3.14152, 2)
#10º desafio

# Qual o erro deste código abaixo?

x = -10

y = 5



# Retorna o menor valor absoluto entre duas variáveis x e y

smallest_abs = min(abs(x,y))
#11º desafio...

# Qual o erro deste código abaixo?

def f(x):

    y = abs(x)

return y



print(f(5))
#12º desafio...

# Segue um exemplo básico de condições:

def inspecao(x):

    if x == 0:

        print(x, "é zero")

    elif x > 0:

        print(x, "é positivo")

    elif x < 0:

        print(x, "é negativo")

    else:

        print(x, "é diferente de tudo que já vi...")



        

# Chama a função

inspecao(0)

inspecao(-12)

inspecao(7)

inspecao('*')

inspecao('Jaques')
#13º desafio...

# Após ( : ) as linhas identadas pertencem ao corpo da função e a linha que não foi identada representa a finalização da função

def f(x):

    if x > 0:

        print("Imprime x se for positivo; x =", x)

        print("Também imprime x se for positivo; x =", x)

    print("Sempre imprime x, independente do seu valor; x =", x)



f(1)

f(-1)

f(0)
#14º desafio...

def teste(nota):

    resultado = 'Falso' if nota < 0 else 'Vardadeiro'

    print('Seu valor é', resultado, 'no teste positivo', nota)

    

teste(45)

teste(-1)

teste(0)
#15º desafio...

dias1 = ['Sábado','Domingo','Segunda', 'Terça','Quarta','Quinta','Sexta']

print ('1º caso')

print (dias1 [3])

print (dias1 [0:4])



# 2ª opcao

dias2 = [

    [2, 'Segunda'],

    [3, 'Terça'],

    [5, 'Quinta']

]

print ('2º caso')    

print (dias2 [2])



dias3 = [[2, 'Segunda'],[3, 'Terça'],[5, 'Quinta']]

print ('3º caso')

print (dias3 [1])
texto = 'Lei Especial de Gratificação ApLicada.'

msg = ''



# O que o código abaixo vai fazer?

for char in texto:

    if char.isupper():

        print(char, end='')
# 1º exercicio

nomes = ['Gates', 'Madog', 'Bob Esponja', 'Fabio Assunção', 'Alan Turing']

for nome in nomes:

    print("Olá ",nome, " teremos uma reunião nesta segunda-feira as 10h.")
