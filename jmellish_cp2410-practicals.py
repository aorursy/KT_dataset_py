#Exercise 1

def is_multiple(n,m):

    return n % m == 0



print(is_multiple(4,2))

print(is_multiple(3,5))
#Exercise 2

print([2 ** i for i in range(0,9)])

#Exercise 3

def all_distinct(seq):

    for i in range(0, len(seq)):

        for x in range(i + 1, len(seq)):

            if seq[i] == seq[x]:

                return False

    return True



print(is_distinct([1,2,3,4,1]))

print(is_distinct([1,2,3,4]))

def harmonic_list(n):

    h = 0

    for i in range(1 , n+1):

        h+= 1/i

        yield h

        

print(harmonic_list(3))