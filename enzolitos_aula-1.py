# Isto aqui é uma célula de Python.

# E a # indica ser um comentário
exemplo = 10 # Exemplo recebe 10

print("O tipo de exemplo é: ", type(exemplo)) # Imprime o tipo de A



exemplo = "ABC" # Exemplo recebe "ABC"

print("O tipo de exemplo agora é: ", type(exemplo)) # Imprime o tipo de A



exemplo = ["oi", 1, 1.54] # Exemplo recebe uma lista

print("O tipo de exemplo finalmente é: ", type(exemplo)) # Imprime o tipo de A
int_example = 11 # Exemplo de int que recebe 11

print("int_example vale", int_example, "e seu tipo é:", type(int_example))



string_example = "exemploooo" # Exemplo de string que recebe "exemploooo"

print("string_example vale", string_example, "e seu tipo é:", type(string_example))



float_example = 11.11 # Exemplo de float que recebe 11.11

print("float_example vale", float_example, "e seu tipo é:", type(float_example))



bool_example = True # Exemplo de boolean que recebe True

print("bool_example vale", bool_example, "e seu tipo é:", type(bool_example))
a = 2

b = 4



print(a + b) # A mais B

print(a - b) # A menos B

print(a * b) # A vezes B

print(a / b) # A dividido por B

print(b % a) # O módulo de b dividido por a

print(a ** b) # A elevado a potência B
string = "abc"

print(string + 1)
string = "abc"

string2 = "def"

print(string2*string)
um = "abc"

dois = "abcdef"



print(um + dois) # Para a soma, tudo ok

print(um - dois) # Agora a subtração não está definida para strings! Por isso, o erro.
a = 1 # a recebe 1

print(a) # Imprimindo o valor de a



a = 2 # a recebe 2

nome = "enzolitos"

print(nome, a) # Imprimindo o valor de a e um parâmetro



a = 3 # a recebe 3

print("O quadrado de a, que vale", a, "é igual a", a**2) # Realizando operações dentro do print
a = 1

b = 2



print("O tipo de a é ", type(a))

print("O tipo de b é ", type(b))



print("O valor de a é igual ao valor de b?", a == b)

print("O tipo de a é igual ao tipo de b?", type(a) == type(b))
valor = 0.1

print(int(valor))
pagou = 1030

custo = 512.65

troco = pagou-custo

print ("o troco deve ser de", troco)



nota100 = 100

nota50 = 50

nota20 = 20

nota10 = 10

nota5 = 5

nota2 = 2

moeda1 = 1

moeda50 = 0.50

moeda25 = 0.25

moeda10 = 0.10

moeda05 = 0.05

moeda01 = 0.01



def calcula_e_imprime_troco(troco_atual, nota):

    num_notas = troco_atual // nota

    print ("quantas notas de", nota, num_notas)

    novo_troco = troco_atual - nota * num_notas

    print ("e agora quanto falta", novo_troco) 

    return novo_troco



notas = [100, 50, 20, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]



for nota in notas:

    troco = calcula_e_imprime_troco(troco, nota)



# troco = calcula_e_imprime_troco(troco, nota100)

# troco = calcula_e_imprime_troco(troco, nota50)

# troco = calcula_e_imprime_troco(troco, nota20)

# troco = calcula_e_imprime_troco(troco, nota10)

# troco = calcula_e_imprime_troco(troco, nota5)

# troco = calcula_e_imprime_troco(troco, nota2)

# troco = calcula_e_imprime_troco(troco, moeda1)

# troco = calcula_e_imprime_troco(troco, moeda50)

# troco = calcula_e_imprime_troco(troco, moeda25)

# troco = calcula_e_imprime_troco(troco, moeda10)

# troco = calcula_e_imprime_troco(troco, moeda05)

# troco = calcula_e_imprime_troco(troco, moeda01)



# notasdecem = troco//nota100

# print ("quantas notas de 100", notasdecem)

# print ("e agora quanto falta", troco - nota100 * notasdecem) 

# troco = troco - nota100 * notasdecem



# notasdecinquenta = troco//nota50

# print ("notas de 50", notasdecinquenta)

# print ("agoooooora falta", troco - nota50 * notasdecinquenta)

# troco = troco - nota50 * notasdecinquenta



# notasdevinte = troco//nota20

# print ("notas de 20", notasdevinte)

# print ("mas agora meu irmão ta faltando", troco - nota20 * notasdevinte)

# troco = troco - nota20 * notasdevinte



# notasdedez = troco//nota10

# print ("notas de 10", notasdedez)

# print ("agora meu anjo lhe faltam", troco - nota10 * notasdedez)

# troco = troco - nota10 * notasdedez



# notasdecinco = troco//nota5

# print ("notas de 5", notasdecinco)

# print ("agora a ultima nota", troco - nota5 * notasdecinco)

# troco = troco - nota5 * notasdecinco



# moedaum = troco//moeda1

# print ("quantas moedas de um", moedaum)

# print ("pague o que falta", troco - moeda1 * moedaum)

# troco = troco - moeda1 * moedaum



# moedacinquenta = troco//moeda50

# print ("quantas moedas de cinquentinha", moedacinquenta)

# print ("ta cabano", troco - moeda50 * moedacinquenta)

# troco = troco - moeda50 * moedacinquenta



# moedavintecinco = troco//moeda25

# print ("quantas vintola", moedavintecinco)

# print ("vamo vamo", troco - moeda25 * moedavintecinco)

# troco = troco - moeda25 * moedavintecinco



# moedadez = troco//moeda10

# print ("dezzzzzzo", moedadez)

# print ("o pouco que lhe falta", troco - moeda10 * moedadez)

# troco = troco - moeda10 * moedadez



# moedacinco = troco//moeda05

# print ("quantas moeda de cincO", moedacinco)

# print ("falta só de ummmmmm", troco - moeda05 * moedacinco)

# troco = troco - moeda05 * moedacinco



# moedahum = troco//moeda01

# print ("quantas moeda de um brother", moedahum)

# print ("ACABOU NÉ", troco - moeda01 * moedahum)
a = 10

b = 5



maior = (a > b)

print(maior)



menor = (a < b)

print(menor)



igual = (a == b)

print(igual)
a = 5



print(a < 10 and a > 1) # and exige que as duas condições sejam verdadeiras

print(a < 3 and a > 1)

print(a < 10 or a > 8) # or exige que apenas uma das condições seja verdadeira

print((a < 3 and a > 1) or (a >= 5 and a**2 < 1000))



print(not True)
# <= - menor igual 

# >= - maior igual

# != - diferente

# is - compara se dois **objetos** são iguais (mais complexo)

# in - compara se existe na coleção
a = 10

if (a < 10):

    print("a é menor que 10")

else:

    print("a não é menor que 10")
a = 9

if (a < 10):

    a = 100

    print("a é menor que 10! mas agora é", a)

else:

    a = 0

    print("a não era menor que 10! mas agora é", a) # Perceba que a identação está errada!

a = 90



# if (a < 90):

#     print("a é menor que 100!")

# elif (a >= 80):

#     print("a é maior ou igual a 80!")

# elif (a < 70):

#     print('abcate')

# else: 

#     print("a é maior que 100")

    

if (a < 100):

    print("a é menor que 100!")  

    

if (a >= 80):

    print("a é maior ou igual a 80!")

else: 

    print("a é maior que 100")