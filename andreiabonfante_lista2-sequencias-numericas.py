n = int(input())

x0 = 1

i = 1



while i <= n:

    xi = i * x0

    x0 = xi   # o novo valor passa a ser guardado

    i = i+1



print(" O fatorial de ",n, "é: ", xi)

    
print(" Entre com um valor de x entre 0 e 2")

x = float(input())

a0 = 1

c0 = 1 - x



ai = a0 * (1 + c0)

ci = c0 ** 2

    

E = abs(ai - a0)

print("Erro encontrado: ", E)



while (E >= (10**(-9))):

    c0 = ci  # guarda em c0 o valor anterior

    a0 = ai

    ai = a0 * (1 + c0)

    ci = c0 ** 2

    E = abs(ai - a0)

    print("Erro encontrado: ", E)



print("Valor 1/x: ",ai)

k = 1.7  # definir aqui o valor da constante k

S = 1.0   # primeiro valor da serie

flag = 1

den = 2

quant = 20



while ((S <= k) and (quant > 0)):

    if (flag == 1):

        S = S + (1/den)

        flag = 0

    else:

        S = S - (1/den)

        flag = 1

    #print("Valor de S até agora:", S)

    den = den + 2

    quant = quant - 1

    

print(" Valor de S: ", S)

    



        

    

    



A = float(input())

x0 = 1

xn1 = (1/2) * (x0 + (A/x0))

E = abs(xn1 - x0)

x0 = xn1





while E >= (10**(-6)):

    xn1 = (1/2) * (x0 + (A/x0))

    E = abs(xn1 - x0)

    x0 = xn1    





print("O valor da raiz é: ", x0)
