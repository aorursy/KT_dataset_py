print('Hello world!')
2 + 2
2 - 2
2 * 2
2/2
2 ** 3
var1 = 1 #--- Inteira

var2 = 1.1 #--- Float

var3 = 'Teste' #--- String

var4 = 'True'
x = 2

y = 3

soma = x + y

x == y #--- x é igual à y : False

x != y #--- x é diferente de y: True
#--- and



x == y and x == soma
#--- or

x == y or x != soma
x = 1

y = 100000



if x > y:

    print('x é maior que y')
if x < y:

    print('x é menor que y')
if x > y:

    print('x é maior que y')

else:

    print('x é menor que y')
x = 1

y = 2



if x == y:

    print('Números iguais')

elif x > y:

    print('x é maior que y')

elif x < y:

    print('x é menor que y')
x = 1



while x < 10:

    print('O número é', x)

    x += 1 #--- Mesma coisa x = x + 1
lista1 = [1, 2, 3, 4, 5]

lista2 = ['Olá', 'Mundo', '!']

lista3 = [0, 'Olá', 'Biscoito', 9.99, True]



for i in lista3:

    print('O elemento é', i)
for i in range(10, 20, 2):

    print('O número é', i)
var1 = 'Hello'

var2 = 'World'



var1 + var2
var1 + ' ' + var2
concatenar = var1 + ' ' + var2



len(concatenar)
var2[3]
concatenar[0:3]
def soma1(a, b):

    print(a + b)





soma1(2, 3)
arquivo = open('../input/Teste1.txt')
arquivo
linhas = arquivo.readline()



print(linhas)
for i in linhas:

    print(i)
texto_completo = arquivo.read()



print(texto_completo)
w = open('arquivo2.txt', 'w')



w.write('Esse é o meu arquivo')



w.close()
lista_frutas = ['abacaxi', 'melancia', 'abacate']



lista_frutas
lista_frutas.append('limao')



lista_frutas
if 'melancia' in lista_frutas:

    print('melancia está na lista')

else:

    print('não está na lista')
del lista_frutas[0:1]



lista_frutas
lista_frutas1 = []



lista_frutas1
lista_frutas1.append(57)



lista_frutas1
lista = [124, 345, 5, 72, 46, 6, 7, 3, 1, 7, 0]



lista.sort()



print(lista)
lista.sort(reverse = True)



print(lista)
lista.reverse()



print(lista)
lista2 = ['bola', 'abacate', 'dinheiro']





lista2.sort()



print(lista2)
import random



numero = random.randint(0, 10)



numero



# Set.seed ::: random.seed(1 ou qualquer outro número)
lista = [6, 45, 9]



n_aleatorio = random.choice(lista)



n_aleatorio
idade = input('Qual a sua idade?')



if float(idade) >= 18:

    print('Maior de idade')

else:

    print('Menor de idade')
nota1 = input('Informe a primeira nota')



if float(nota1) >= 6:

    print('Aprovado')

else:

    print('Reprovado')
nota2 = input('Informe a segunda nota')



if float(nota2) >= 6:

    print('Aprovado')

else:

    print('Reprovado')
#--- Importando a função sqrt da lib math



from math import sqrt



a = input('Informe o valor de a')

b = input('Informe o valor de b')

c = input('Informe o valor de c')



a = float(a)

b = float(b)

c = float(c)



delta = b ** 2 - (4 * a * c)



print('O valor de delta é', delta)



x1 = (-b + sqrt(delta))/2*a

x2 = (-b - sqrt(delta))/2*a



print('A primeira raiz é', x1)

print('A segunda raiz é', x2)





#--- Nota: Deveria ter feito um if para não deixar ele pegar valores negativos, pq o sqrt não 

#--- deixa



#--- Exemplo: a = 1, b = -10 e c = 24
n1 = float(input('Informe o primeiro valor da lista'))

n2 = float(input('Informe o segundo valor da lista'))

n3 = float(input('Informe o terceiro valor da lista'))



lista = [n1, n2, n3]



lista.sort()



print(lista)
n1 = input('Informe o primeiro número')

n2 = input('Informe o segundo número')

sinal = input('Informe um sinal')



n1 = float(n1)

n2 = float(n2)





if sinal == 'soma':

    print(n1 + n2)

elif sinal == 'subtração':

    print(n1 - n2)

elif sinal == 'multiplicação':

    print(n1 * n2)

elif sinal == 'divisão':

    print(n1/n2)
