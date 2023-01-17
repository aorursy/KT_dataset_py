a = 10



# Usando a função print:

print(a)



# Usando a função type:

type(a)



# Usando ambas juntas 

print(type(a))



# Uma outra maneira:

tipo_a = type(a)

print("Segunda forma: ", tipo_a)
# Definindo a função

def funcao_soma(parametro1, parametro2): # Parâmetros passados: (...)

    soma = parametro1 + parametro2 # Bloco de código

    return soma # Retorna a variável



# Declarando duas variáveis

a = 10 

b = 20



# Chamando nossa função e colocando seu retorno em soma

soma = funcao_soma(a, b)

print(soma)    
# Versão sem função - tudo faz parte do código principal

a = 10

b = 20

c = 30

soma = 0

for i in range(0, c):

    # Se número par, soma a*i - b

    if (c%2 == 0):

        soma = soma + (a*i)-b

    # No outro caso, subtrai de soma b*i

    else:

        soma = soma - b*i

print("Versão sem função:", soma)



# ----------------------------------



# Versão com função

# Função de soma_alternada que recebe, a, b e executa a soma alternada c vezes

def soma_alternada(a, b, c):

    soma = 0

    for i in range(0, c):

    # Se número par, soma a*i - b

        if (c%2 == 0):

            soma = soma + (a*i)-b

        # No outro caso, subtrai de soma b*i

        else:

            soma = soma - b*i

    return soma



# Código principal

a = 10

b = 20

c = 30

print("Versão com função:", soma_alternada(a, b, c))



# Perceba que podemos mudar os parâmetros sem dificuldade:

print("Versão com função, outros valores:",soma_alternada(5, 4, 100))



# E até gerar gráficos interessantes com facilidade

import matplotlib.pyplot as plt

c = range(0, 100)

x = [soma_alternada(10, 10, i) for i in c]

plt.plot(c, x)
# Criando uma função que calcula quatro operações básicas de matemática para dois números:

def calculadora(a, b):

    return a+b, a-b, a*b, a/b



a = 5

b = 6

soma, subtracao, produto, quociente = calculadora(5, 6)

print("Soma:", soma, " / Subtração:", subtracao, "/ Produto", produto, "/ Quociente", quociente)



# -----------------------------------------------



# Definindo valores padrões para os parâmetros

def porcentagem(a, b=100): #Aqui definimos que, caso a pessoa não passe b, ele é obrigatoriamente 100. Mas ainda existe a possibilidade do usuário mudar.

    return(a/b*100)



a = 5

print(a, " é ", porcentagem(a),"% de 100")

b = 6

print(a, " é ", porcentagem(a, b), "% de", b)
# Lembre-se que na função acima declaramos primeiro a, depois b

print(porcentagem(b=200, a=100))
# Função que recebe uma lista e seta o índice 1 como valendo 1

def set_1_as_1(lista):

    lista[1] = 1



# Criamos a lista 1

lista = [5, 4, 3]



# Lista2 recebe lista 1

lista2 = lista

print(lista)

print(lista2)



# Alteramos o primeiro índice da lista

set_1_as_1(lista)

print(lista)



# Observe que a lista2 muda também. a explicação para isso é como um ponteiro funciona.

#Vou tentar explicar resumidamente abaixo

print(lista2)
# Para usar módulos e bibliotecas, o processo é:



# Importar a biblioteca

import math



# Usar as funções

print("Raiz:", math.sqrt(10)) # Precisamos do "math."



# Em alguns casos, podemos importar uma única função

from math import sin



print("Seno:", sin(5)) # Nesse caso, não precisamos do "bamath."
