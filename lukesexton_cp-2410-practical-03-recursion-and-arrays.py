def maximum_element(sequence, start_number):

    #build base case

    if start_number > len(sequence):

        return "Number exceeds maximum elements of the sequence, try again."

    elif start_number == len(sequence):

        return start_number

    else:

        return maximum_element(sequence, start_number + 1)
sequence = [1,3,5,7,9]

start_number = 6

test = maximum_element(sequence, start_number)

print(test)
def product(m, n):

    if n == 1:

        return m

    else:

        return m + product(m, n - 1)
test3 = product(2, 3)

print(test3)