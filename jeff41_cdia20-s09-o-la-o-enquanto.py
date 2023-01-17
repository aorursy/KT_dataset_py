# soma todos os inteiros entrados por um usuário

def main():
    soma = 0
    msg = "Entre com um inteiro (Digite zero para sair): "
    num = int( input(msg) )
    while (num != 0):
        soma += num
        num = int( input(msg) )
    print("A soma de todos esses numeros eh {}".format(soma))
    
main()
# Encontra o menor divisor de um número

def divide(k, n):
    # Retorna True se k divide n
    return n % k == 0

def min_div(n):
    # Encontra o menor divisor (além do 1) para qualquer n > 1
    k = 2
    while (k < n) and not divide(k, n):
        k += 1
    return k

def main():
    n = int( input("Entre com um n > 1: ") )
    print("O menor divisor de {} eh {}.".format( n, min_div(n) ))

main()
# Imprime os primos entre 2 e 100

def eh_primo(n):
    # Retorna True se n eh primo
    return n > 1 and min_div(n) == n

def main():
    n = 100
    for n in range(2, n + 1):
        if (eh_primo(n)):
            print(n, end=" ")
    print()

main()