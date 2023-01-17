## Importa os pacotes preliminares 

import matplotlib

import numpy as np
def bisec(erro, k, a, b, F, casas):

    

    if F(a) == None:

        print("ERRO: não existe uma função") #ativa caso o usuário não defina uma função antes de chamar bissec()

        return



    if F(a)*F(b) > 0: #condição do teorema para que o método convirja é F(a)*F(b) <= 0 

        print("esse intervalo não apresenta as condições para se usar o método, o intervalo deve obedecer F(a)*F(b) <= 0")

        return



    i = 0

    x = (a+b)*0.5



    x = iteracao_bissec(erro, k, a, b, F, i, x)

    x = round(x, casas)

    return x
def iteracao_bissec(erro, k, a, b, F, i, x):

    if i > k: #verificação se já bateu o número de iterações

        print("número maximo de iterações feitas")

        return x



    if ((erro*a)**2) > ((a - x)**2): #verificação se já está abaixo do erro

        print("convergiu")

        return x



    if F(x)*F(b) < 0: #verificação se x vai substituir a ou b

        return iteracao_bissec(erro, k, x, b, F, i+1 , (x+b)/2)

    else:

        return iteracao_bissec(erro, k, a, x, F, i+1 , (x+a)/2)
def funcao(x):

     return (x**2 - 1)



print(funcao(0))

print(funcao(3))
bisec(0.001,100,0,3,funcao,3)