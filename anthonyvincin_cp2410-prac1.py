def is_multiple(n, m):

    return True if n % m == 0 else False





print(is_multiple(10, 3))

print(is_multiple(9, 3))

number_list = [pow(2, i) for i in range(0, 9)]

print(number_list)

def is_distinct(numbers):

    if len(numbers) == len(set(numbers)):

        return True

    else:

        return False





print(is_distinct([1, 2, 2, 3]))

print(is_distinct([1, 2, 3, 4]))

def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h





my_list = list(harmonic_gen(3))

print(my_list)
