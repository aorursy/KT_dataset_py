import numpy as np
import pandas as pd
import random
import seaborn as sns
def statistcs(numbers):
    media = np.mean(numbers)
    moda = np.bincount(numbers).argmax()
    mediana = np.median(numbers)
    variancia = np.var(numbers)
    desvio_padrao = np.sqrt(variancia)
    coeficiente = (desvio_padrao / media) * 100
    
    data = {
        "Numeros": numbers,
        "Média": round(media,4),
        "Moda": round(moda,4),
        "Mediana": round(mediana,4),
        "Variancia": round(variancia,4),
        "Desvio Padrão": round(desvio_padrao,4),
        "Coeficiente de Variação": round(coeficiente,4)
    }

    return data


numbers = [1,3,5,7,9,5,45,50,35, 40, 22, 12]
    
a = statistcs(numbers)
a


##read file

def read_file(file):
    file_object = open(file, "r")
    lines = file_object.readlines()  
    numbers = []  
    
    for line in lines:
        number = list(map(int, line.split(",")))
        numbers.append(number)

    return numbers
#must have 1 double value
def check_duplicated_number(numbers):
    for number in numbers:
        if numbers.count(number) > 1:
            return True
    return False
#must have at least 6 unique values
def check_duplicated_numbers_limit(numbers):
    LIMIT = 6
    count = 0
    for number in numbers:
        if numbers.count(number) == 1:
            count = count + 1
    
    if count >= LIMIT:
        return True
    else:
        return False
def get_statistics_classroom_from_file(file):
    classroom = read_file(file)
    return get_statistcs_classroom(classroom)    
def get_statistcs_classroom(classroom):
    data = []
    for student in classroom:
        if not check_duplicated_number(student):
            print("Não possui um numero repetido na lista.")
            continue
        if not check_duplicated_numbers_limit(student):
            print("Não possui quantidade minima de números únicos.")
            continue
        data.append(statistcs(student))
    return data
file = "../input/fpe-inputs/list.txt"
a = get_statistics_classroom_from_file(file)
print(a)
#create numbers for a student
def create_numbers():
    numbers = list(np.random.uniform(low=1, high=50, size=11))
    numbers.append(np.median(numbers))
    numbers.sort()
    return numbers
#validate authenticity
def validade_authenticity(classroom_numbers, student_numbers):
    for element in classroom_numbers:
        if element == student_numbers:
            return False
    return True
#create classroom
def create_classroom(size):
    classroom = []
    count = 0
    while (count < size):
        student = create_numbers()
        if validade_authenticity(classroom, student):
            count = count + 1
            classroom.append(student)
    return classroom
#statistics for classroom
classroom = create_classroom(50)
classroom_data = get_statistcs_classroom(classroom)

df = pd.DataFrame(classroom_data)

df
sns.kdeplot(df.Média)
