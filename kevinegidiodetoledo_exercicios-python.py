print("Olá, mundo")
nome = input("Óla, para podermos começar, por favor, me diga o seu nome: ")
print(" ")
print("Ólaaa {}, seja muito bem vindo(a)".format(nome))

n_1 = float(input("por favor me diga o primeiro numero: "))
n_2 = float(input(" agora me diga o segundo numero: "))
soma = n_1 + n_2
print(" a soma dos dois numeros é {}".format(soma))
a = input("digite algo: ")
print("Seu numero primiticvo é: ", type(a))
print("Só tem espaço? ", a.isspace())
print("É um numero? ", a.isnumeric())
print("É alfabetico? ", a.isalpha())
print("É alfanumerico? ", a.isalnum())
print("É maiúscula? ", a.isupper())
print("É minúscula? ", a.islower())

a = float(input("me diga o seu número: "))
b = a - 1
c = a + 1
print(" ")
print("O sucessor do seu numero é {} e o seu antecessor é {}".format(c,b))


num = int(input("digite um numero: "))
dobro = num * 2
triplo = num * 3
raiz = num ** (1/2)
print(" ")
print("O dobro de {} é {}".format(num,dobro))
print(" ")
print("O triplo de {} é {}".format(num,triplo))
print(" ")
print("A raiz quadrada de {} é {}".format(num,raiz))


nome = input("qual seria o nome do aluno? ")
print(" ")
nota_1 = float(input("diga a primeira nota do {}: ".format(nome)))
nota_2 = float(input("Diga a segunda nota do {}: ".format(nome)))
media = (nota_1 + nota_2)/2
print(" ")
print("A media do aluno {} foi {}".format(nome,media))
m = int(input("Por favor me diga o seu numero(em metros): "))
cm = m * 100
mm = m * 1000
print(" ")
print("{}m vale {}cm".format(m,cm))
print(" ")
print("{}m vale {}mm".format(m,mm))
num = int(input("digite um numero: "))
print(" ")
mult = int(input("Gostaria de chegar até qual lugar na tabuada?"))
rang = (num * mult) + 1
for i in range(0,rang,num):
 print(i)
din = float(input("quanto de dinheiro você gostaria de calcular? "))
dol = din/4.90
print(" ")
print("Você tem exatamente {:.2f} dólares".format(dol))

lar = float(int(input("diga a largura da sua parede: ")))
alt = float(int(input("agora diga a altura da sua parede: ")))
metro_quadradro = lar * alt
print(" ")
print("Sua parede tem {:.1f}m²".format(metro_quadradro))
print(" ")
quantidade_tinta = metro_quadradro/2
print("Você precisará de {:.1f}L de tinta".format(quantidade_tinta))


val = float(input("digite o valor do produto: "))
des = float(input("agora me diga a procentagem de desconto no produto: "))
val_fin = (val - ((val * des)/100))
print(" ")
print("O valor do produto com o desconto será {:.2f}".format(val_fin))
atual = int(float(input("Digite o valor sálarial atual do funcionário: ")))
por = int(float(input("Agora nos diga a quantidade de aumento(em porcentagem) que o novo funcionário receberá: ")))
print(" ")
aum = (atual + ((atual * por)/100))
dife = aum - atual
print("O novo salário do funcionário será de {:.2f} com uma diferença de {:.2f} para o antigo".format(aum, dife))
valor = float(int(input("diga o valor da temperatura: ")))
temp = input("O valor está em qual escala termométrica? (C/F/K): ")
print(" ")
cf = (((valor * 9)/5) + 32)
ck = valor + 273.15
if temp == "C":
        print("O valor em fahrenheit será {:.2f} e valerá {:.2f} em Kelvin".format(cf,ck))


fc = (((valor - 32) * 5)/9)    
fk = ((((valor - 32) * 5)/9) + 273.15)
if temp == "F":
    print("O valor em Celsius será {:.2f} e valerá {:.2f} em Kelvin".format(fc,fk))


kf = ((((valor - 273.15) * 9)/5) + 32)
kc = valor - 273.15
if temp == "K":
    print("O valor em fahrenheit será {:.2F} e valerá {:.2F} em Celsius.".format(kf,kc))
      
        
dias = int(input("Quantos dias ficará com o carro? "))
preç_dia = float(input("Preço por dia? "))
print(" ")
km = int(float(input("Quantos km rodados? ")))
preç_km = float(input("Preço por km rodado? "))
preço = (km * preç_km) + (dias * preç_dia)
print(" ")
print("Você terá que pagar um total de {:.2f} R$".format(preço))

from math import trunc
num = float(input("digite um valor: "))
print(trunc(num))
cato = float(input("Valor do cateto oposto: "))
cata = float(input("Valor do cateto adjacente: "))
hip = (cato**2)+(cata**2)
hip_fin = hip**(1/2)
print(" ")
print("O valor da hipotenusa é igual a {:.2f}".format(hip_fin))
print("Para que possamos descobrir os valores de seno, cosseno e a tangente, por favor me diga os valores dos catetos.")
cato = float(input("Valor do cateto oposto: "))
cata = float(input("Valor do cateto adjacente: "))
hip = (cato**2)+(cata**2)
hip_fin = hip**(1/2)
print(" ")
sen = cato/hip_fin
cos = cata/hip_fin
tan = cato/cata
print("O seno é {:.3f}, o cosseno é {:.3f} e a tangente é {:.3f}".format(sen,cos,tan))




    


import random
qnt = int(input("Quantos alunos tem na sala? "))
for i in range(1,qnt + 1):
    aluno = input("nome aluno {}: ".format(i))
sorteado = random.choice(aluno)
print(" ")
print("E o aluno sorteado foi {}".format(sorteado))
import random
qnt = int(input("Quantos grupos tem na sala? (maximo de 5): "))

if qnt == 2:
    gp1 = input("nome grupo 1: ")
    gp2 = input("nome grupo 2: ")
    lista = [gp1,gp2]
    random.shuffle(lista)
    print("E a ordem será: ")
    print(lista)
    
if qnt == 3:
    gp1 = input("nome grupo 1: ")
    gp2 = input("nome grupo 2: ")
    gp3 = input("nome grupo 3: ")
    lista = [gp1,gp2,gp3]
    random.shuffle(lista)
    print("E a ordem será: ")
    print(lista)

if qnt == 4:
    gp1 = input("nome grupo 1: ")
    gp2 = input("nome grupo 2: ")
    gp3 = input("nome grupo 3: ")
    gp4 = input("nome grupo 4: ")
    lista = [gp1,gp2,gp3,gp4]
    random.shuffle(lista)
    print("E a ordem será: ")
    print(lista)

if qnt == 5:
    gp1 = input("nome grupo 1: ")
    gp2 = input("nome grupo 2: ")
    gp3 = input("nome grupo 3: ")
    gp4 = input("nome grupo 4: ")
    gp5 = input("nome grupo 5: ")
    lista = [gp1,gp2,gp3,gp4,gp5]
    random.shuffle(lista)
    print("E a ordem será: ")
    print(lista)

import pygame
from pygame.locals import *
pygame.init()
arq = input("me diga o nome do arquivo: ")
pygame.mixer.music.load(arq)
pygame.music.play()
pygame.event.wait()
nome = str(input("Por favor me diga seu nome: "))
print(" ")
print("seu nome em letras minúsculas {}".format(nome.upper()))
print(" ")
print("seu nome em letras maiúsculas {}".format(nome.lower()))
print(" ")
separa = nome.split()
print("seu nome {} tem {} de letras".format(nome, len(nome) - nome.count(' ')))
print(" ")
print("seu primeiro nome é {} e ele tem {} letras".format(separa[0], len(separa[0])))
from time import sleep

número = int(input("Por favor me diga um número de 0 a 9999: "))
num = str(número)
n = len(num)
print("Só um minuto estamos processando o número.")
sleep(len(num))
print(" ")
if n == 1:
    print("Unidade: {}".format(num))

if n == 2:
    print("Unidade: {}".format(num[0]))
    print("Dezena: {}".format(num))
    
if n == 3:
    print("Unidade: {}".format(num[0]))
    print("Dezena: {}{}".format(num[0],num[1]))
    print("Centena: {}".format(num))
    
if n == 4:
    print("Unidade: {}".format(num[0]))
    print("Dezena: {}{}".format(num[0],num[1]))
    print("Centena: {}{}{}".format(num[0],num[1],num[2]))
    print("Milhar: {}".format(num))

nome = input("Diga o nome da cidade: ")

n = nome.split()
a = n[0]

if a == "Santos":
    print("Começa com Santos")
    
else:
    print("Não começa com Santos")

nome = input("Por favor me diga o seu nome: ")
n = nome.split()
nom = len(n)

for i in range(0, nom):
    if n[i] == "Silva":
        print("O seu nome tem Silva na palavra {}!!, muito bom".format(i))
    if n[i] != "Silva":
        print("A palavra {} do seu nome não é silva".format(i))
    print(" ")
nome = input("Diga uma frase: ")
print("A letra (a) aparece {} vezes na sua frase".format(nome.count('a')))
lista = list(nome)
no = [list(palavra) for palavra in lista]
n = len(no)
listaa = []
for i in range(0,n):
    if no[i] == ['a'] or no[i] == ['A']:
        listaa.append(i)
print("A letra a ou A aparece a primeira vem na posição {} e a ultima na posição {}".format(listaa[0], listaa[len(listaa)-1]))
nome = input("me diga o seu nome por favor: ")


print(" ")
sim_nao = input("Seu nome completo é {}? (sim/nao): ".format(nome))
    
while sim_nao == "nao":
    nome = input("me diga o seu nome por favor: ")
    sim_nao = input("Seu nome completo é {}? (sim/nao): ".format(nome))


if sim_nao == "sim":
    print(' ')
    nom = nome.split()
    print("Nome completo: {}".format(nome))
    print("Seu primeiro nome é: {}".format(nom[0]))
    le = len(nom)- 1
    print("Seu último nome é: {}".format(nom[le]))
from time import sleep
import random



name = input("Olá amigo(a), tudo bem? espero que simm, então, antes de podermos começar nossa brincadeira, você poderia me falar o seu nome? só o primeiro nome já está bom: ")
sleep(2.0)
print("aaa {} que nome lindo, bem vamos começar então.".format(name))
print(' ')
sleep(1.0)
print("Pense em um número {}.".format(name))
print(" ")
sleep(5.5)
print("Pensou? agora guarde esse número, pois eu vou tentar adivinhar ele!!")
sleep(1.0)
primeira_pergunta = input("O numero que você pensou é maior que 50?(sim/nao): ")

###########################################################################################################################################################################################

if primeira_pergunta == "sim":
    sleep(2.0)
    print(' ')
    print("hummmm, imaginei que você pensaria em um número a cima de 50 {}".format(name))
    sleep(1.0)
    print("Então me diga".format(name))
    sleep(1.0)
    segunda_pergunta = input("Seu número está 51 e 70?(sim/nao): ")

############################################################################################################################################################################################    
#25
    if segunda_pergunta == "sim":
        print(' ')
        sleep(1.0)
        print("Entendiii, muito bem {}, então me diga".format(name))
        sleep(1.0)
        terceira_pergunta = input("Seu número é menor ou maior que 61?(maior/menor): ")

###########################################################################################################################################################################################        

        if terceira_pergunta == "menor":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número emmm".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)
            quarta_pergunta = input("seu número está entre 51 e 56?(sim/nao): ")


###########################################################################################################################################################################################            

            if quarta_pergunta == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 51 e 56, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando = random.randint(51,56)
                print("E o número sorteado foooi {}".format(rando))
                sleep(0.5)

 ###########################################################################################################################################################################################                    

                resultado = input("era esse o número?(sim/nao): ")
                if resultado == "sim":
                    sleep(1.0)
                    numero_errado  = rando - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################################

            if quarta_pergunta == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 57 e 60, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_2 = random.randint(57,60)
                print("E o número sorteado foooi {}".format(rando_2))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_2 = input("era esse o número?(sim/nao): ")
                if resultado_2 == "sim":
                    sleep(1.0)
                    numero_errado_2  = rando_2 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_2))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_2 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))    

###############################################################################################################################################################################/        

        if terceira_pergunta == "maior":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número.".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)
            quarta_pergunta_2 = input("seu número está entre 61 e 65?(sim/nao): ")

#########################################################################################################################################################################            

            if quarta_pergunta_2 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 62 e 65, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                print(" ")
                rando_3 = random.randint(62,65)
                print("E o número sorteado foooi {}".format(rando_3))
                sleep(0.5)

#########################################################################################################################################################################            

                resultado_3 = input("era esse o número?(sim/nao): ")
                if resultado == "sim":
                    sleep(1.0)
                    numero_errado_3  = rando_3 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_3))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_3 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

###############################################################################################################################################################################            


            if quarta_pergunta_2 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 66 e 70, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_4 = random.randint(66,70)
                print("E o número sorteado foooi {}".format(rando_4))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_2 = input("era esse o número?(sim/nao): ")
                if resultado_2 == "sim":
                    sleep(1.0)
                    numero_errado_4  = rando_4 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_4))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_2 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))


 ###############################################################################################################################################################################       
         ###
    if segunda_pergunta == "nao":
        print(' ')
        sleep(1.0)
        print("Entendiii, muito bem {}, então me diga".format(name))
        sleep(1.0)

#########################################################################################################################################################################        

        terceira_pergunta_3 = input("Seu número é menor ou maior que 86?(maior/menor): ")
        if terceira_pergunta_3 == "menor":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número emmm".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)

#########################################################################################################################################################################            

            quarta_pergunta_3 = input("seu número está entre 71 e 78?(sim/nao): ")
            if quarta_pergunta_3 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 71 e 78, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                    print(" ")
                rando_5 = random.randint(71,78)
                print("E o número sorteado foooi {}".format(rando_5))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_3 = input("era esse o número?(sim/nao): ")
                if resultado_3 == "sim":
                    sleep(1.0)
                    numero_errado_5  = rando_5 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_5))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_3 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################            

            if quarta_pergunta_3 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 79 e 86, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_6 = random.randint(79,86)
                print("E o número sorteado foooi {}".format(rando_6))
                sleep(0.5)

#########################################################################################################################################################################            

                resultado_4 = input("era esse o número?(sim/nao): ")
                if resultado_4 == "sim":
                    sleep(1.0)
                    numero_errado_6  = rando_6 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_6))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_4 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################        
          ##########      
        if terceira_pergunta_3 == "maior":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número.".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)

#########################################################################################################################################################################            

            quarta_pergunta_4 = input("seu número está entre 87 e 93?(sim/nao): ")
            if quarta_pergunta_4 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 87 e 93, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")

#########################################################################################################################################################################                

                rando_7 = random.randint(87,95)
                print("E o número sorteado foooi {}".format(rando_7))
                sleep(0.5)
                resultado_5 = input("era esse o número?(sim/nao): ")

#########################################################################################################################################################################                

                if resultado_5 == "sim":
                    sleep(1.0)
                    numero_errado_7  = rando_7 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_7))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_5 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################            

            if quarta_pergunta_4 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 94 e 100, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")

#########################################################################################################################################################################                

                rando_8 = random.randint(94,100)
                print("E o número sorteado foooi {}".format(rando_8))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_6 = input("era esse o número?(sim/nao): ")
                if resultado_6 == "sim":
                    numero_errado_8  = rando_8 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_8))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_6 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("mas hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))


                    
#########################################################################################################################################################################    
#########################################################################################################################################################################    
#########################################################################################################################################################################    
########################################################################################################################################################################


if primeira_pergunta == "nao":
    sleep(2.0)
    print(' ')
    print("hummmm, imaginei que você pensaria em um número a baixo de 50 {}".format(name))
    sleep(1.0)
    print("Então me diga".format(name))
    sleep(1.0)
    segunda_pergunta_2 = input("Seu número está 26 e 50?(sim/nao): ")

############################################################################################################################################################################################    
#25
    if segunda_pergunta_2 == "sim":
        print(' ')
        sleep(1.0)
        print("Entendiii, muito bem {}, então me diga".format(name))
        sleep(1.0)
        terceira_pergunta_5 = input("Seu número é menor ou maior que 38?(maior/menor): ")

###########################################################################################################################################################################################        

        if terceira_pergunta_5 == "menor":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número emmm".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)
            quarta_pergunta_9 = input("seu número está entre 26 e 32?(sim/nao): ")


###########################################################################################################################################################################################            

            if quarta_pergunta_9 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 26 e 32, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_9 = random.randint(26,32)
                print("E o número sorteado foooi {}".format(rando_9))
                sleep(0.5)

 ###########################################################################################################################################################################################                    

                resultado_9 = input("era esse o número?(sim/nao): ")
                if resultado_9 == "sim":
                    sleep(1.0)
                    numero_errado_9  = rando_9 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_9))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_9 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print(" mas ta bom, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################################

            if quarta_pergunta_9 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 33 e 38, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_10 = random.randint(33,38)
                print("E o número sorteado foooi {}".format(rando_10))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_10 = input("era esse o número?(sim/nao): ")
                if resultado_10 == "sim":
                    sleep(1.0)
                    numero_errado_10  = rando_10 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_10))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_10 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))    

###############################################################################################################################################################################/        

        if terceira_pergunta_5 == "maior":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número.".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)
            quarta_pergunta_11 = input("seu número está entre 39 e 44?(sim/nao): ")

#########################################################################################################################################################################            

            if quarta_pergunta_11 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 39 e 44, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                print(" ")
                rando_11 = random.randint(39,44)
                print("E o número sorteado foooi {}".format(rando_11))
                sleep(0.5)

#########################################################################################################################################################################            

                resultado_11 = input("era esse o número?(sim/nao): ")
                if resultado_11 == "sim":
                    sleep(1.0)
                    numero_errado_11  = rando_11 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_11))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_11 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

###############################################################################################################################################################################            


            if quarta_pergunta_11 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 45 e 50, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_12 = random.randint(45,50)
                print("E o número sorteado foooi {}".format(rando_12))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_12 = input("era esse o número?(sim/nao): ")
                if resultado_12 == "sim":
                    sleep(1.0)
                    numero_errado_12  = rando_12 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_12))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_2 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))


 ###############################################################################################################################################################################       
         ###
    if segunda_pergunta_2 == "nao":
        print(' ')
        sleep(1.0)
        print("Entendiii, muito bem {}, então me diga".format(name))
        sleep(1.0)

#########################################################################################################################################################################        

        terceira_pergunta_13 = input("Seu número é menor ou maior que 12?(maior/menor): ")
        if terceira_pergunta_13 == "menor":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número emmm".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)

#########################################################################################################################################################################            

            quarta_pergunta_13 = input("seu número está entre 0 e 6?(sim/nao): ")
            if quarta_pergunta_13 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 0 e 6, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                    print(" ")
                rando_13 = random.randint(0,6)
                print("E o número sorteado foooi {}".format(rando_13))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_13 = input("era esse o número?(sim/nao): ")
                if resultado_13 == "sim":
                    sleep(1.0)
                    numero_errado_13  = rando_13 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_13))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_3 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca")
                    print("ah mas ta bom, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################            

            if quarta_pergunta_13 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 7 e 12, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")
                rando_14 = random.randint(7,12)
                print("E o número sorteado foooi {}".format(rando_14))
                sleep(0.5)

#########################################################################################################################################################################            

                resultado_14 = input("era esse o número?(sim/nao): ")
                if resultado_14 == "sim":
                    sleep(1.0)
                    numero_errado_14  = rando_14 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_14))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_14 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("ah mas ta bom foi quaseeee hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################        
          ##########      
        if terceira_pergunta_13 == "maior":
            sleep(2.0)
            print("olha só {} eu acho que já estou quaseeee adivinhando seu número.".format(name))
            sleep(0.5)
            print("agora me diga {}".format(name))
            sleep(0.5)

#########################################################################################################################################################################            

            quarta_pergunta_15 = input("seu número está entre 13 e 18?(sim/nao): ")
            if quarta_pergunta_15 == "sim":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 13 e 18, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")

#########################################################################################################################################################################                

                rando_15 = random.randint(13,18)
                print("E o número sorteado foooi {}".format(rando_15))
                sleep(0.5)
                resultado_5 = input("era esse o número?(sim/nao): ")

#########################################################################################################################################################################                

                if resultado_15 == "sim":
                    sleep(1.0)
                    numero_errado_15  = rando_15 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_15))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_15 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca")
                    print("mas ta bom hahaha, na próxima eu acerto pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))

#########################################################################################################################################################################            

            if quarta_pergunta_15 == "nao":
                sleep(2.0)
                print("Olha só {} eu não tenho certeza sobre o número que você pensou, mas vou tentar chutar algo entre 19 e 25, quem sabe eu acerto né.".format(name))
                sleep(1.0)
                print("Só pra deixar mais empolgante, vou fazer uma contagem regressiva hahaha")
                sleep(4.0)
                for cont in range(3,0,-1):
                    print(cont)
                    sleep(1.0)
                print(" ")

#########################################################################################################################################################################                

                rando_16 = random.randint(19,25)
                print("E o número sorteado foooi {}".format(rando_16))
                sleep(0.5)

#########################################################################################################################################################################                

                resultado_16 = input("era esse o número?(sim/nao): ")
                if resultado_16 == "sim":
                    numero_errado_16  = rando_16 - 1
                    print("hahaha olha vou ser sincera, eu quase falei {} sorte que eu mudei de idéia na hora hahah".format(numero_errado_16))
                    sleep(1.0)
                    print("Bem {} foi muito legal ter você por aqui viu? até a próximaaa.".format(name))


                if resultado_16 == "nao":
                    sleep(1.5)
                    print("Aaaaaah não brinca.")
                    print("mas hahaha, na próxima eu acerto, pode ter certeza viu!!!")
                    sleep(1.0)
                    print(" ")
                    print("Bem {}, eu vou ficando por aqui tá? foi muito bom ter você por aquiii, volte mais vezes tá? beijooos".format(name))
a = float(input("Por favor me diga o valor de a: "))
b = float(input("Por favor me diga o valor de b: "))
c = float(input("Por favor me diga o valor de c: "))

delta = (b**2)-(4*a*c)

print(" ")
print("O valor de Delta é: {}".format(delta))

if delta > 0:
    baskhar = ((-b) + delta)/(2*a)
    baskhar_2 = ((-b) - delta)/(2*a)
    print("o x1 = {} //// e o x2 = {}".format(baskhar,baskhar_2))
    
if delta < 0:
    print("Esse número não contem baskhar, pois seu valor é negativo")
    
if delta == 0:
    baskhar_3 = ((-b) + delta)/(2*a)
    baskhar_4 = ((-b) - delta)/(2*a)
    print("o x1 = {} //// e o x2 = {}".format(baskhar_3,baskhar_4))
    
import random

print("Eu iria penar em um numero de 0 a 5 e duvidooo você adivinhar qual foi o numero que eu pensei.")
print(" ")
numero = random.randint(0,5)

resposta = int(input("Qual numero você pensou? "))

if resposta == numero:
    print("Parabeeeeens você acertou!! eu pensei exatamento no numero {}".format(numero))
    
    sn = input("gostaria de tentar de novo? (s/n): ")
    
    while sn == "s":
        print("Eu iria penar em um numero de 0 a 5 e duvidooo você adivinhar qual foi o numero que eu pensei.")
        print(" ")
        
        numero = random.randint(0,5)
        resposta = int(input("Qual numero você pensou? "))
        
        if resposta == numero:
            print("Parabeeeeens você acertou!! eu pensei exatamento no numero {}".format(numero))
            sn = input("gostaria de tentar de novo? (s/n): ")
            
            if sn == "n":
                print("ook entãoo, ate a proxima")
                
        if resposta != numero:
            print("Não foi dessa vez!! eu pensei exatamento no numero {} não no {}".format(numero, resposta))
            
            sn = input("gostaria de tentar de novo? (s/n): ")
            
            if sn == "n":
                print("ook entãoo, ate a proxima")
    
    
    if sn == "n":
        print("ook entãoo, ate a proxima")
        
    
    
    
    
else:
    print("Não foi dessa vez!! eu pensei exatamento no numero {} não no {}".format(numero, resposta))
    
    sn = input("gostaria de tentar de novo? (s/n): ")
    
    while sn == "s":
        print("Eu iria penar em um numero de 0 a 5 e duvidooo você adivinhar qual foi o numero que eu pensei.")
        print(" ")
        numero = random.randint(0,5)
        resposta = int(input("Qual numero você pensou? "))
        
        if resposta == numero:
            print("Parabeeeeens você acertou!! eu pensei exatamento no numero {}".format(numero))

            sn = input("gostaria de tentar de novo? (s/n): ")
            
            if sn == "n":
                print("ook entãoo, ate a proxima")
                            
        if resposta != numero:
            print("Não foi dessa vez!! eu pensei exatamento no numero {} não no {}".format(numero, resposta))
            
            sn = input("gostaria de tentar de novo? (s/n): ")
            
            if sn == "n":
                print("ook entãoo, ate a proxima")
    if sn == "n":
        print("ook entãoo, ate a proxima")


    
    
velocidade = float(input("Qual a velocidade do carro? "))

if velocidade > 80.00:
    multa = (80 - velocidade) * 7
    print("PARADO, Você levou uma Multa de {} reais".format(multa))
    
if velocidade < 80.00:
    print("pode ir, você está dentro dos limites de velocidade")
numero = float(input("Qual o numero desejado? "))

if numero % 2 == 1:
    print("esse numero não é impar")

if numero % 2 == 0:
    print("esse numero é par")
print("Olá, fico feliz que escolheu nossa companhia para viagens, aqui nos temos uma promoção onde cobramos R$0,50 por Km para viagens de até 200Km e R$0,45 parta viagens mais longas, espero que aproveite nossa promoção, a companhia kevin agradece.")
print(" ")
distancia = float(input("Qual a distancia entre o seu local atual e o destino? "))

if distancia > 200.00:
    print(" ")
    valor = distancia * 0.45
    print("O valor da viagem ficará {}".format(valor))
    
    
    
if distancia < 200.00:
    print(" ")
    valor = distancia * 0.50
    print("O valor da viagem ficará {}".format(valor))
valor_casa = float(input("Me diga o valor da casa: "))
salário = float(input("Agora me diga o seu salário por favor: "))
parcelamento = int(input("Em quantos anos você pretende parcelar: "))

prestação_mensal = (parcelamento *12) / valor_casa

porcentagem_salario = 0.3 * salário

if prestação_mensal > porcentagem_salario:
    print("O empréstimo foi aceito.")
    
else:
    print("O empréstimo foi negado")
