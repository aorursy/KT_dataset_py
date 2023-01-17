# Faça o código da primeira questão aqui

print("Entre com um número inteiro positivo que representa os segundos a serem convertidos em horas,minutos e segundos")

n = int(input())



h = n // 3600;   # parte inteira da divisao



resto = (n % 3600)     # resto da divisao



m = resto // 60



s = (resto % 60)



print(n," segundos representa ",h," horas, ", m, " minutos e ", s, " segundos")

# Questao 2



print("Calculando y = f(x) + g(x) para x = 1,2,...10")

x = 1

while (x <= 10):

    hx = (x*x) - 16

    if (hx >= 0):

        fx = hx

    else:

        fx = 1

    if (fx == 0):

        gx = (x*x) + 16

    else:

        gx = 0

    y = fx + gx 



    print("==========")

    print("x = ",x)

    print("hx = ",hx)

    print("fx = ",fx)

    print("gx = ",gx)

    print("y = ",y)

    print("==========")

    

    x = x + 1



# Questão 3



print("Conversão decimal binário")

print("Entre com número decimal (maior ou igual a 1) a ser convertido para binário (-1 pára)")



n = int(input())



while (n >= 1):  # checa se o numero é valido

    print("numero decimal: ", n)

    print("numero binario.. as impressoes devem ser lidas de baixo para cima:")

    auxiliar = n

    while (auxiliar != 1):

        resto = auxiliar % 2

        auxiliar = auxiliar // 2

        print(resto)

    print(auxiliar)

    print("Entre com número decimal (maior ou igual a 1) a ser convertido para binário (-1 pára)")

    n = int(input())



    

# Questão 4

# verificar a biblioteca decimal

# from decimal import Decimal

A = 0.0

seno = 0.0

print("Angulo A:",A,"-- Seno:",seno)



# comeca a calular a partir do angulo A = 0,1

A = 0.1    

while (A < 6.3):

    erro = 1.0

    fatA = 1

    exp = 1

    sinal = +1

    seno = 0.0

    while (erro >= 10**(-35)):

        print(" seno parcial ", seno)

        seno_anterior = seno

        seno = seno + (((A**exp) * sinal) / fatA) # acumulando os elementos da serie

        erro = abs(seno-seno_anterior)

        print("erro parcial ", erro)

        sinal = sinal * (-1)  # atualiza sinal para que da proxima vez ele seja diferente

        

        # calculando o proximo A

        auxiliar = exp

        exp = exp + 2

        while(auxiliar <= exp):

            fatA = fatA * auxiliar

            auxiliar = auxiliar + 1

        #print("calculo do fatorial de ",exp," = ",fatA)

    print("Angulo A:",A,"-- Seno:",seno)

    print(" =========== pegando proximo A ============")

    A = A + 0.1  # pega proximo A para ser calculado o seno

    

    

    

    
