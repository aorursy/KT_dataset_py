def prime_number(number):
    if not isinstance(number, int):
        return 'invalid argument'
    if number < 0:
        return "invalid argument"
    if number > 5.5 * (10**7):
        return "argument out of range"
    
    if number > 1:
        for i in range(2,number):
            if number % i == 0:
                return "not a prime number"
        
        return "prime number"
    else:
        return "not a prime number"
print(prime_number(409))
print(prime_number((5.5 * (10**7)) + 1))
print(prime_number(407))
print(prime_number('abc'))
def sum_of_products(list1, list2):
    def check_is_number(list):
        for i in list:
            if not isinstance(i, int) and not isinstance(i, float):
                return False
        return True
    
    def products_elements(biggerList, smallerList):
        for i in range(0, len(biggerList)):
            try:
                productsSum.append(biggerList[i] * smallerList[i])
            except:
                productsSum.append(biggerList[i])
        return productsSum
    
    if len(list1) == 0 and len(list2) == 0:
        return -1
    
    if check_is_number(list1) and check_is_number(list2):
        productsSum = []
        
        if len(list1) > len(list2):
            productsSum = products_elements(list1, list2)
        else:
            productsSum = products_elements(list2, list1)
        
        return sum(productsSum)
    else:
        return "wrong number"
    
print(sum_of_products([10,20,30,40], [2,3,4]))
print(sum_of_products([1,2,3,4,5,6], [7,8,9,10,11,12]))
print(sum_of_products([], []))
print(sum_of_products([1,2,3], [2,3,'']))
def growth_rate(population_a, population_b):
    if not isinstance(population_a, int) or not isinstance(population_b, int):
        return "invalid argument"
    elif population_a <= 0 or population_b <= 0:
        return "invalid argument"
    
    years = 0
    
    while population_a < population_b:
        years += 1
        population_a += population_a * 0.03
        population_b += population_b * 0.015
        
    return years
print(growth_rate(90000.123, 200000))
print(growth_rate(90000, -20))
print(growth_rate(90000, 200000))
import numpy as np

def count_list(numberList):
    d = {}
    
    d['max'] = max(numberList)
    d['sum'] = sum(numberList)
    d['occurs'] = numberList.count(numberList[0])
    d['mean'] = np.mean(numberList)
    
    nearestList = []
    for i in numberList:
        nearestList.append(i - d['mean'])
        
    d['near-mean'] = nearestList.index(min(nearestList))
    d['minus'] = sum(i for i in numberList if i < 0)
    
    return d
print(count_list([1, 2, 7, 20, -10, 15, -20, 1, 30, 1]))
