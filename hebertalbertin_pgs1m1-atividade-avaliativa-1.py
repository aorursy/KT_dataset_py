from collections import Counter
import numpy as np
import pandas as pd
import random
def check_same_value(lista):
    # Pelo menos um valor deve estar repetido.
    
    c = Counter(lista)

    for key, counts_value in list(c.items()):
        if counts_value < 2:
            del c[key]

    return len(c) >= 1
def check_unique_values(lista):
    # O conjunto de dados deve apresentar, pelo menos, 6 valores únicos.
    
    c = Counter(lista)
    wrong_list = []
    for key, counts_value in list(c.items()):
        if counts_value > 6:
            wrong_list.append(c[key])
    
    return not len(wrong_list) >= 1
def check_not_equals(lista):
    # NÃO É PERMITIDO escolher todos os números iguais.
    
    c = Counter(lista)
    wrong_list = []
    for key, counts_value in list(c.items()):
        if counts_value == 12:
            wrong_list.append(c[key])
    
    return not len(wrong_list) >= 1
def validate_numbers(lista):
        return check_same_value(lista) and check_unique_values(lista) and check_not_equals(lista)
def calculate_statistics(lista):
    if validate_numbers(lista):
        l_pd = pd.Series(lista)

        media = np.average(lista)
        counts = l_pd.value_counts()
        moda = counts[counts == max(counts)]
        moda = [m for m in moda.index]
        moda.sort()
        mediana = l_pd.median()

        desvio = np.std(lista)

        variancia = desvio**2

        coefiente = (desvio/media) * 100

        print('Lista:',  lista)
        print('\nMédia:', format(media, '.4f'))
        print("Moda:", end = ' ') 
        print(*moda, sep = ", ")
        print('Mediana:',  format(mediana, '.4f'))
        print('Variância amostral:',  format(variancia, '.4f'))
        print('Desvio padrão amostral:', format(desvio, '.4f'))
        print('Coeficiente de variação:', format(coefiente, '.4f'))
        print('\n-------------------------\n')
    else:
        print('Lista inválida!')
        print('\n-------------------------\n')
def read_file(file):
    with open(file, 'r') as f:
        listas = []
        for line in f:
            numeros = line.split(', ')
            lista = [int(n) for n in numeros]
            lista.sort()
            
            listas.append(lista)
        return listas
# valid_list = read_file('../input/12-numeros-por-linha/12-numeros-por-linha.txt')
listas = read_file('../input/12-numeros-por-linha/12-numeros-por-linha.txt')

for lista in listas:
    calculate_statistics(lista)
def generate_random_singular():
    max_numbers = 12
    max_value = 50
    
    lista = []
    for x in range(0, max_numbers):
        # se for o ultimo a inserir, insere o anterior gerado para termos um repetido
        if x == max_numbers-1:
            lista.append(random_number)
        else:
            random_number = random.randint(1,max_value)
            lista.append(random_number)
    lista.sort()
    
    # valida as regras
    if validate_numbers(lista):
        return lista
    else:
        return generate_random_singular()
    
def generate_random_lists():
    quantity = 50
    
    listas = []
    
    for x in range(0, quantity):
        lista = generate_random_singular()
        listas.append(lista)
    return listas
def check_identicals(listas):
    # Trabalho Individual
    
    for lista in listas:
        lista.sort()
    
    for z in range(0, len(listas)):
        for x in range(0, len(listas)):
            if not z == x:
                if listas[z] == listas[x]:
                    return x
    return -1
# gerar listas válidas
listas = generate_random_lists()
# validar se nenhuma é igual
valid = False
while not valid:
    wrong_index = check_identicals(listas)
    
    if wrong_index == -1:
        valid = True
    else:
        print('Lista no índice ' + str(wrong_index) + ' trocado')
        del listas[wrong_index]
        listas.append(generate_random_singular())

print('Listas válidas geradas:', listas)
# estatísticas para todas as listas
for lista in listas:
    calculate_statistics(lista)
