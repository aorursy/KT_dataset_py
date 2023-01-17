import math

math.factorial(100)
def fact_sum_digits(num):

    fact_num=math.factorial(num)

    sum_digits=0

    

    while(fact_num > 0):

        remainder = fact_num % 10

        sum_digits += remainder

        fact_num = fact_num // 10

        

    return sum_digits

        

print(fact_sum_digits(100))