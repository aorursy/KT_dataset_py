def mul(num):
    """
    Prints the multipliaction table of a given number
    """
    for i in range(1, 11):
        print("{multiplier} * {multiplicand} = {multiplicantion}".format(
            multiplier=num, multiplicand=i, multiplicantion=num * i))

mul(9)
def checkPrime(max_num):
    """
    Check whether the given number is prime or not
    """
    for num in range (2, max_num):
        if max_num % num == 0:
            return False
    return True

def twinPrime(max_num):
    """
    Generates the list of twin primes
    """
    for first_num in range(2, max_num):
        second_num = first_num + 2
        if (checkPrime(first_num) and checkPrime(second_num)):
            print(" {0} and {1}".format(first_num, second_num))

print("Twin Prime: ")
twinPrime(1000)
import math

prime_list = []

def primeFactors(num):
    """
    Returns the prime factors of a number
    """
    
    # for and while loop takes care of composite numbers
    while num % 2 == 0:
        prime_list.append(2)
        num = num/2
        
    # num will be odd by now, thus complexity can be reduced by discarding even numbers
    # sqrt is used to discard composite numbers
    for i in range(3, int(math.sqrt(num))+1, 2):
        while num%i == 0:
            prime_list.append(i)
            num = num/i

    # when num is prime
    if num > 2:
        num = int(num)
        prime_list.append(num)
    return prime_list
        
primeFactors(56)
import operator as op

def factorial(num):
    """
    Returns the factorial of a number
    """
    
    if num == 1:
        return num
    return num * factorial(num-1)

def permutation(n, r):
    """
    Returns the permutation of a number
    """
    return int(factorial(n) / factorial(n-r))

def combination(n, r):
    """
    Returns the combinations of a number
    """
    return int(factorial(n) / (factorial(r) * factorial(n-r)))

print("Permutation: ", permutation(15,4))
print("Combination: ", combination(15,4))
def decToBin(num):
    """
    Prints the binary number of a given decimal number using recursion
    """
    if num > 1:
        decToBin(num//2)
    print(num % 2, end="")
        
decToBin(11)
def exp(num, power):
    """
    Returns the exponent of a given number with power
    """
    if power == 0:
        return 1
    if num == 0:
        return 0
    answer = num
    increment = num
    
    for i in range(1, power):
        for j in range(1, num):
            answer += increment
        increment = answer
    return answer

def cubesum(num):
    """
    Returns the sum of cube of each digit of a given number
    """
    sum = 0
    temp = num
    while temp > 0:
        digit = temp % 10
        sum += exp(digit, 3) # digit ** 3
        temp //= 10
    return sum

def isArmstrong(num):
    """
    Check whether given number is Armstrong or not
    """
    if cubesum(num) == num:
        return True
    else:
        return False

def printArmstrong(num):
    """
    Prints armstrong number in given range
    """
    armstrong = []
    for n in range(num):
        if isArmstrong(n):
            armstrong.append(n)
    return armstrong
    
print("Is 371 an armstrong number: ", isArmstrong(371))
print("Armstrong: ", printArmstrong(1000))
def prodDigit(num):
    """
    Returns the product of digits of given number
    """
    temp = num
    prod = 1
    while temp > 0:
        digit = temp % 10
        prod *= digit
        temp //= 10
    return prod


# num = int(input("Enter a number: "))
num = 54
print("Product of all digits of {0} is: {1}".format(num, prodDigit(num)))
def MDR(num):
    """
    Returns the MDR (Multiplicative Digital Root and Multiplicative Persistance of the given number
    """
    s = str(num)
    pers = 0
    while len(s) > 1:
        s = str(prodDigit(int(s)))
        pers += 1
    return int(s), pers

num = 341
mdr, mper = MDR(num)
print("For {0} MDR is {1} and M Persistance is {2}".format(num, mdr, mper))
def sumPdivisors(num):
    """
    Returns the sum of proper divisors of a number
    """
    divisors = []
    for i in range(1, num):        
        if num % i == 0:
#             print("{0} is divisible by {1}".format(num, i))
            divisors.append(i)
    return sum(divisors)
            
            
sumPdivisors(36)
def isPerfect(num):
    """
    Checks whether given number is perfect or not
    """
    return num == sumPdivisors(num)

def perfectNums(lower, upper):
    """
    Prints the all perfect numbers in given range
    """
    for i in range(lower, upper):
        if isPerfect(i):
            print(i)
            
perfectNums(0,100)
def amicableNum(lower, upper):
    """
    Prints all amicable numbers in given range
    """
    for num in range(lower, upper+1):
        for num_ in range(num, upper+1):
            if num != num_:
                if amicablePair(num, num_):
                    print(num, num_)
        
def amicablePair(num1, num2):
    """
    Checks whether given pair is amicable or not
    """
    return (sumPdivisors(num1) == num2) and (sumPdivisors(num2) == num1)

amicableNum(1, 1000)
def filterOdd(lst):
    """
    Filter odd numbers from given list
    """
    return list(filter(lambda num: (num%2 != 0), lst))

filterOdd([0,2,5,8,19,20,34,95])
def cube(lst):
    """
    Returns the list of cubes of given number
    """
    return list(map(lambda x: x**3, lst))

cube([1, 3, 5, 9, 15])
def evenCube(lst):
    """
    Returns the even cubes from the given list of numbers
    """
    return cube(list(filter(lambda num: (num%2) == 0, lst)))

evenCube([0,2,5,8,19,20,34,95])