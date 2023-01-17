# Question 1

def is_multiple(n,m):

    return n % m == 0 



# Test

print(is_multiple(4,2))

print(is_multiple(3,2))
# Question 2

[2 ** i for i in range (0,9)]
# Question 3

def unique_list(numbers):

    for i in range(len(numbers)):

        if numbers.count(numbers[i]) > 1:

            return False

    return True



# Test

print(unique_list([1,3,7,5]))

print(unique_list([1,1,7,5]))
# Question 4

def harmonic_list(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h



# Test

for harmonic in harmonic_list(5):

     print(harmonic)
