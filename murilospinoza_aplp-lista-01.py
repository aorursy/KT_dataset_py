import math

def prime_number(number):

    # Verifica se o número não é positivo     

    if number <= 0:

        return 'invalid argument'



    # Verifica se o número é maior que o limite

    if number > 55000000:

        return 'argument out of range'



    # Verifica se o número é 1 ou é par e maior que 2

    # Para não precisar testar se ele é divisível pelos números pares até ele

    if number == 1 or (number > 2 and number % 2 == 0):

        return 'not a prime number'



    # Testa os números impares de 3 até o número informado

    for n in range(3, round(math.sqrt(number) + 1), 2):

        if number % n == 0:

            return 'not a prime number'



    return 'prime number'
print(prime_number(-10))

print(prime_number(1))

print(prime_number(6))

print(prime_number(11))

print(prime_number(75629))

print(prime_number(55000001))
def sum_of_products(listaA, listaB):

    # Valida se pelo menos uma das listas tem algum elemento

    if len(listaA) == 0 and len(listaB) == 0:

        return -1

    

    # Deixa a variável listaA com a maior lista

    if len(listaA) < len(listaB):

        aux = listaA

        listaA = listaB

        listaB = aux

    

    # Inicia a variável de somatória

    produtos = 0

    

    # Percorre os índices da lista A

    for i in range(len(listaA)):

        # Recebe o valor A

        a = listaA[i]



        b = 1

        # Define o valor B, se a listaB possuir o índice desejado

        if i < len(listaB):

            b = listaB[i]

        

        # Verifica se os dois valores são numéricos

        if not ((isinstance(a, int) or isinstance(a, float)) and (isinstance(b, int) or isinstance(b, float))):

            return 'wrong number'

        

        # Calcula o produto e soma ao agregado

        produtos += a * b



    return produtos
# {}

print( sum_of_products([], []) )



# 6 + 30 + 17 = 53

print( sum_of_products([2,5], [3,6,17]) )



# wrong

print( sum_of_products([2,5], [2, 'wrong',17]) )
def growth_rate(population_a, population_b):

    # Varifica se os valores são inteiros e maiores que zero

    if isinstance(population_a, int) and population_a > 0 and isinstance(population_b, int) and population_b > 0:

        # Define a taxa de crescimento da população A e B

        growthA = 1.03

        growthB = 1.015

        

        # Cotnador de anos

        years = 0



        while population_a < population_b:

            # Avança um ano e aumenta a população

            years += 1

            population_a = int(population_a * growthA)

            population_b = int(population_b * growthB)

        return years

    else:

        return 'invalid argument'
# POPULAÇÕES IGUALADAS

# A = 100 e B = 102 => 1 ano => A = 103 e B = 103

print( growth_rate(100,102) )



# PAÍS A ULTRAPASSANDO

# A = 90000000

# B = 200000000

# + 55 anos

# A = 457393321

# B = 453588751

print( growth_rate(90000000, 200000000) )
import numpy as np

def count_list(numbers):

    # Média

    mean = np.mean(numbers)



    # Número mais próximo à média

    nearMean = min(numbers, key=lambda item : abs(item - mean))



    # Procura os números negativos

    negatives = list(filter(lambda number: number < 0, numbers))

    

    # Constrói o dicionário

    dados = {

        'max': max(numbers),

        'sum': sum(numbers),

        'occurs': numbers.count(numbers[0]),

        'mean': mean,

        'near-mean': nearMean,

        'minus': sum(negatives)

    }

    return dados
numbers = [1, 3, 8, 1, -5]

print(count_list(numbers))