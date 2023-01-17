#Escreva as funções F(n) e G(n) de acordo com as fórmulas

def F(n):

    if n == 1:

        n = 2

    elif n == 2:

        n = 1

    else:

        n = 2*F(n-1) + G(n-2)

    return n     



def G(n):

    if n >= 3:

        n = G(n-1) + 3*F(n-2)

    return n
#Escreva uma função K(n) que receba um inteiro n > 0 e devolva uma tupla com F(n) e G(n)

def K(n):

    if(n > 0):

        return (F(n), G(n))
#Para k=2, a função deve devolver os valores 1 e 2

print(K(2))



#Para k=3, a função deve devolver os valores 3 e 8

print(K(3))



#Para k=4, a função deve devolver os valores 8 e 11

print(K(4))
#Imprima uma lista de resultados (F(n), G(n)) para K entre 1 e 5, usando List Comprehension

[K(n) for n in range(1, 6)]
#Reescreva as funções usando lambdas



F_lambda = lambda n: 1 if n == 2 else 2 if n == 1 else 2*F_lambda(n-1) + G_lambda(n-2)

G_lambda = lambda n: G_lambda(n-1) + 3*F_lambda(n-2) if n >= 3 else n

K_lambda = lambda n: None if n < 0 else (F_lambda(n), G_lambda(n))



[K_lambda(n) for n in range(1, 6)]