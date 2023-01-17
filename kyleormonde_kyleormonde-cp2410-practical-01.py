# Question 1

def is_multiple(n, m):

    if m % n == 0:

        return True

    else:

        return False





print("Question 1:\n")

print(is_multiple(10, 10), "\n")

print(is_multiple(11, 10))
# Question 2

def squares_list(i):

    list_size = 9

    my_list = [2 ** i for i in range(0, list_size)]

    print("Question 2:\n", my_list, "\n")





squares_list(2)
# Question 3

def distinct(x, y, z):

    number_list = [x, y, z]

    number_set = set(number_list)



    if len(number_set) < len(number_list):

        return "Indistinct"

    else:

        return "Distinct"





print("Question 3:\n")

print(distinct(1, 2, 2), "\n")

print(distinct(1, 2, 3))

# Question 4

def harmonic_factors(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h





print("Question 4:")

for index in harmonic_factors(20):

    print(float(index))