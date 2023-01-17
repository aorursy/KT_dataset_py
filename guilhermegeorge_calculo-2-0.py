from sympy import *

from time import sleep



#Declaração Simbolica para realizar as funçoes de integral multiplicação etx

x = Symbol('x')

A1 = Symbol('A1')

A2 = Symbol('A2')

A3 = Symbol('A3')

A4 = Symbol('A4')

A5 = Symbol('A5')





pretty_print("Olá, Algoritimo para calcular integral por frações parciais tipo 1 e tipo 2")

sleep(1)

pretty_print("No formato 2*x**2")

pretty_print(Integral(2*x**2))

sleep(1)

pretty_print("Insira o numerador e denominador separadamente")

sleep(1)



# Função Para resolver equaçoes do tipo 1

def tipo1(lst,stA):

    for i in range(len(a)):



        if a[i] != '*' or a[i] == '*' and a[i + 1] == 'x':

            stA = stA + a[i]



        if a[i] == '*' and a[i + 1] != 'x':

            if a[i + 1] != 'x':

                if stA != "":

                    lst.append(stA)

                    stA = ""



        if i == len(a) - 1:

            if stA != "":

                lst.append(stA)

    for j in range(len(lst)):

        base.append(lst[j])

    for h in range (len(base)):

        base[h] = sympify (base[h])





#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



# Função Para resolver equaçoes do tipo 2





def tipo2(lst,stA):

    for i in range(len(a)):



        if a[i] != '*' or a[i] == '*' and a[i + 1] == 'x':

            stA += a[i]



        if a[i] == '*':

            if a[i + 1] != 'x':

                if stA != "":

                    lst.append(stA)

                    stA = ""

        if i == len(a) - 1:

            if stA != "":

                lst.append(stA)

    

    base.append(lst[0])



    for i in range(len(lst)):



        if lst[i] == '2' or lst[i] == '3' or lst[i] == '4':

            Xesque = int(lst[i])

            guarda = i + 1



            for j in range(Xesque):

                J = j + 1

                if J != 1:

                    J = str(J)

                    J = '**' + J

                    BLABLA = lst[i - 1] + J

                    base.append(BLABLA)

                    print("")



                for W in range(guarda, len(lst)):

                    base.append(lst[W])

    apagapo = False

    guarda_A = 0

    for A in range(len(base)):

        for B in range(len(base)):

            if base[A] == base[B] and A != B:

                guarda_A = B

                apagapo = True

                break

    if apagapo:

        del base[guarda_A]

    for j in range(len(base)):

        base[j] = sympify(base[j])







#Ler a função do usuario ou tester

P = expand(input("Digite o numerador "))

Q = expand(input("Digite o denominador "))



print("")

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



#Chamada da Função grau para descobrir se é propria ou impropria

gP = degree(P)



gQ = degree(Q)



if gQ > gP:

    print("Função Própria")

    propria = True

    sleep(1.5)

else:

    print("Função Imprópria")

    sleep(1.5)

    F, R = div(P, Q)



# E informar ao usuario e no caso de impropria realizar conta e calcular o resto





#Exibir como a integral Bonitinha pro usuario

pretty_print(Integral(P / Q))

sleep(1.5)

#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



#Função da biblioteca para encontrar o quadrado redutivel das funções usadas no caso 1 e caso 2

w = factor(Q)



#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



# A String que recebera

a = str(w)

# O vetor de STRING que ira separar os fatores redutiveis em diferentes espaços para elaboração do sistema

lst = []

# A string auxiliar pra preencher o vetor lst a partir da String variavel "a"

stA = ""

# O vetor simbolico que recebera o vetor de string e ira realizar as contas de fato

base = []



# O controle para ver se o fator linear se repete e determinar se é tipo 1 ou tipo 2

Repetidos = False



for w in range (len(a)):

    if a[w] == '*' and a[w-1] == '*':

        Repetidos = True



if Repetidos:

    print("Caso tipo 2")

    sleep(1.5)

    print("")

    tipo2(lst,stA)



else:

    print("Caso do tipo 1")

    sleep(1.5)

    print("")

    tipo1(lst,stA)





A = (A1 / base[0])

Coontrole_BLA = 1

if Coontrole_BLA < len(base):

    A = A + (A2 / base[1])

    Coontrole_BLA = Coontrole_BLA + 1





if Coontrole_BLA < len(base):

    A = A + (A3 / base[2])

    Coontrole_BLA = Coontrole_BLA + 1

if Coontrole_BLA < len(base):

    A = A + (A4 / base[3])

    Coontrole_BLA = Coontrole_BLA + 1



if Coontrole_BLA == 2:

    sistema = ((A1 * base[1]) + (A2 * base[0])) / (base[0] * base[1])

    cima = (A1 * base[1]) + (A2 * base[0])



if Coontrole_BLA == 3:

    sistema = ((A1 * base[1] * base[2]) + (A2 * base[0] * base[2]) + (A3 * base[0] * base[1])) / (base[0] * base[1] * base[2])

    cima = (A1 * base[1] * base[2]) + (A2 * base[0] * base[2]) + (A3 * base[0] * base[1])



if Coontrole_BLA ==4:

    sistema = ((A1 * base[1] * base[2] * base[3]) + (A2 * base[0] * base[2] *base[3]) + (A3 * base[0] * base[1] *base[3]) + (A5 * base[0]* base[1] * base[2] * base[3])) / (base[0] * base[1] * base[2] * base[3])



if gQ < gP:

    pretty_print(factor(sistema))

    sleep(1)

    print("")

    print("")

    print(cima,"=",R)

    sleep(1)

else:

    print(cima,"=",P)



INTEGRALZADA = integrate(P/Q)



print("")

TEST =(Derivative(INTEGRALZADA).doit())



print("")

pretty_print(TEST)

sleep(1.5)

print("")

pretty_print(Integral(TEST))

print("")

pretty_print(INTEGRALZADA)




