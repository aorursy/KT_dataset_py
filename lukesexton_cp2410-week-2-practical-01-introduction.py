def is_multiple(n, m):

    return n % m == 0
test = is_multiple(6, 3)

print(test)
powers_of_two = [2**i for i in range(9)]

print(powers_of_two)
def is_different(numbers):

    if len(set(numbers)) == len(numbers):

        return True

    else:

        return False
numbers = [1,2,3,4,5]

test = is_different(numbers)

if test == True:

    print("Each number is different! :)")

else:

    print("There are duplicates!")
duplicate_numbers = [1,2,3,3,4,5]

test_two = is_different(duplicate_numbers)

if test_two == True:

    print("Each number is different! :)")

else:

    print("There are duplicates!")
def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h
test_harmonic = harmonic_gen(3)

for i in test_harmonic:

    print(i)