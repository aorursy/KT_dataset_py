#Escreva uma função que calcule um determinado elemento da sequencia de Fibonacci, dado sua posição na sequencia

def f(n):

    if n > 1:

        n = f(n-1) + f(n-2)

    return n



f(20)
#Escreva uma expressão lambda que faça a mesma operação

fl = lambda n: fl(n-1) + fl(n-2) if n > 1 else n



fl(20)
#Aplique as funções acima até a sequencia 20 e imprima na tela a lista de valores da sequencia

print('função'.ljust(10) + 'lambda'.ljust(10))

for i in range(1,21):

    print(str(f(i)).ljust(10) + str(fl(i)).ljust(10))
