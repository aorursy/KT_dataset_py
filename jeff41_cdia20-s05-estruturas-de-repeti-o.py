for i in range(10):
    print("olá mundo")
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
for i in range(10):
    print("olá mundo")
    print("hello world")
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
for i in range(10):
    print("olá mundo")
for i in range(10):
    print("hello world")
list( range(10, 2, -1) )
for i in [8, 3, 0]:
    print(i, i ** 2)
lista = [8, 3, 0]
for i in lista:
    print(i, i ** 2)
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
k = 0
for i in range(11):
    k = k + i

print("O valor final do acumulador eh: ", k)
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
k = 1
for i in range(1,11):
    k = k * i

print("O valor final do acumulador de multiplicacao eh: ", k)
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
int("5")
# Listagem 4.1

def harmonica(n):
    # Computa a soma de 1/k para k=1 a n
    total = 0
    for k in range(1, n + 1):
        total += 1 / k
    return total

def main():
    # "5" + 2 
    n = int(input('Entre com um inteiro positivo: '))
    print("A soma de 1/k para k = 1 a", n, "eh", harmonica(n))

main()
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#(espaço reservado para ilustrar a sequência de comandos)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Em um arquivo chamado citacao.py escreva:

def main():
    citacao = input("Entre com uma citacao: ")
    for i in range(1000): # a variavel i varia de 0 a 999
        print(citacao)

# Não se esqueça de chamar a função main() como abaixo
main()
# Em um arquivo chamado potencia.py escreva:

def main():
    # Note que não há necessidade de se obter nada do usuário porque 
    # o enunciado estabelece um intervalo fixo
    
    n = 10
    for i in range(1,n+1):
        print(i, 2**i)

# Não se esqueça de chamar a função main() como abaixo
main()
# Em um arquivo chamado tabela.py escreva:

from math import log

def main():
    # Note que não há necessidade de se obter nada do usuário porque 
    # o enunciado estabelece um intervalo fixo
    
    n = 200
    for i in range(10,n+1,10):
        print("{}\t{}\t{}\t{}\t{}".format(i, log(i), n*log(i), i**2,  2**i))

# Não se esqueça de chamar a função main() como abaixo
main()
# Em um arquivo chamado tabela_circulo.py escreva:

from math import pi

def area_circulo(r):
       return pi * (r ** 2)

def main():
    n = 10
    for r in range(1,n+1):
        print("{}\t{}".format(r, area_circulo(r)))

# Não se esqueça de chamar a função main() como abaixo
main()      
# Em um arquivo chamado soma_pares.py escreva:

def soma_pares(n):
    soma = 0 
    for i in range(1,n+1):
        soma += i*2
    return soma

def main():
    n = input("Entre com um numero natural n: ")
    n = int(n)
    print("A soma dos {} primeiros pares eh {}".format(n,soma_pares(n)))
    
main()