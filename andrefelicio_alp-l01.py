import math

def prime_number(number):
    
    #must be integer
    if not isinstance(number, int):
        return 'invalid argument'
    
    #must be positive
    if not number > 0:
        return 'invalid argument'
    
    #must be lesser than 5.5 * 10^7
    if number > 5.5 * (10 ** 7):
        return 'argument out of range'
    
    #corner case (1 is not a prime number)
    if number == 1:
        return 'not a prime number'
    
    #given an input number n, check whether any integer m from 2 to √n evenly divides n 
    sqrt = int(math.sqrt(number))
    for m in range(2, sqrt+1):
        if number % m == 0:
            return 'not a prime number'
        
    return 'prime number'

print(prime_number(1))
print(prime_number(-2))
print(prime_number('ax'))
print(prime_number(17))
fucking_big_number = int(5.5 * (10 ** 8))
print(prime_number(fucking_big_number))
import numpy as np

def sum_of_products(a, b):
    
    size_a = len(a)
    size_b = len(b)
    
    #if empty lists returns -1
    if size_a == 0 and size_b == 0:  
        return -1
    
    #all elements must be number
    if wrong_number(a) or wrong_number(b):
        return 'wrong number'
    
    #if the different sizes
    if size_a < size_b:
        #padding a
        a += [1] * (size_b - size_a)
    elif size_b < size_a:
        #padding b
        b += [1] * (size_a - size_b)  
       
    #multiply elements with numpy and sum all the products
    return sum(np.array(a) * np.array(b))  
    
    
def wrong_number(l):
    #if the list has a element that not is int or float return false
    return any((not isinstance(e, int) and not isinstance(e, float)) for e in l )
           
a = [1,5,2222]
b = [3,1,17]

sum_of_products(a,b)
def growth_rate(population_a, population_b):
    
    #must be numeric
    if not isinstance(population_a, int) or not isinstance(population_b, int):
        return 'invalid argument'
    
    #must be positive
    if not population_a > 0 or not population_b > 0:
        return 'invalid argument'
    
    #growth
    growth_a = 0.03
    growth_b = 0.015
    
    years = 0
    
    #while A is less than B 
    while population_a <= population_b:
        #increase the population
        population_a = int(population_a * (1 + growth_a))    
        population_b = int(population_b * (1 + growth_b))
        years += 1
    
    return years
a = 90000000
b = 200000000
growth_rate(a,b)
import numpy as np

def count_list(l):
    
    #create a numpy array
    l_array = np.array(l)
    
    dic = {'max': max(l)}
    dic['sum'] = sum(l)
    dic['occurs'] = l.count(l[0])
    dic['mean'] = np.mean(l_array)
    #calcule absolute value of element - mean and get the index of the minimum value
    dic['near-mean'] = l_array[(np.abs(l_array- dic['mean'])).argmin()] 
    #mask to negative numbers then sum
    dic['minus'] = ((l_array < 0)*l_array).sum()
    
    return dic
    
l = [9,5,9,10,-10,-4,8,-3,2,7,4]

print(count_list(l))
