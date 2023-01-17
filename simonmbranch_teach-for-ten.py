%%time

# In order for Kaggle to time our program, we need to add %%time to the start of our program

def sieve(list_of_numbers):

    is_prime = [True for every_number in list_of_numbers]

    for index in range(len(list_of_numbers)):

        for test_index in range(0, index):

            if list_of_numbers[index] % list_of_numbers[test_index] == 0:

                is_prime[index] = False

    return [list_of_numbers[i] for i in range(len(list_of_numbers)) if is_prime[i]]



print(sieve(range(2, 5001)))
%%time

def sieve(list_of_numbers):

    is_prime = [True for every_number in list_of_numbers]

    for index in range(len(list_of_numbers)):

        for test_index in range(index+1, len(list_of_numbers)):

            if list_of_numbers[test_index] % list_of_numbers[index] == 0:

                is_prime[test_index] = False

    return [list_of_numbers[i] for i in range(len(list_of_numbers)) if is_prime[i]]



print(sieve(range(2, 5001)))
%%time

def sieve(list_of_numbers):

    is_prime = [True for every_number in list_of_numbers]

    for index in range(len(list_of_numbers)):

        if is_prime[index] is False:

            # Skip composite numbers

            continue

        for test_index in range(index+1, len(list_of_numbers)):

            if list_of_numbers[test_index] % list_of_numbers[index] == 0:

                is_prime[test_index] = False

    return [list_of_numbers[i] for i in range(len(list_of_numbers)) if is_prime[i]]



print(sieve(range(2, 5001)))
%%time

def sieve(list_of_numbers):

    is_prime = [True for every_number in list_of_numbers]

    for index in range(len(list_of_numbers)):

        if (list_of_numbers[index] ** 2) > list_of_numbers[-1]:

            break

        if is_prime[index] is False:

            # Skip composite numbers

            continue

        for test_index in range(index+1, len(list_of_numbers)):

            if list_of_numbers[test_index] % list_of_numbers[index] == 0:

                is_prime[test_index] = False

    return [list_of_numbers[i] for i in range(len(list_of_numbers)) if is_prime[i]]



print(sieve(range(2, 5001)))
%%time

def sieve(list_of_numbers):

    is_prime = [True for every_number in list_of_numbers]

    for index in range(len(list_of_numbers)):

        if (list_of_numbers[index] ** 2) > list_of_numbers[-1]:

            break

        if is_prime[index] is False:

            # Skip composite numbers

            continue

        index_squared = index ** 2

        for test_index in range(index+1, len(list_of_numbers)):

            if test_index < index_squared:

                continue

            if list_of_numbers[test_index] % list_of_numbers[index] == 0:

                is_prime[test_index] = False

    return [list_of_numbers[i] for i in range(len(list_of_numbers)) if is_prime[i]]



print(sieve(range(2, 5001)))