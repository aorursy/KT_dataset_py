# Caleb Black (JC341205) Prac 2 Submission
# Question 1



def is_multiple(n, m):

    return n % m == 0



assert is_multiple(20, 5)

assert not is_multiple(21, 5)



print('done')
# Question 2



def powers_of_two():

    return [2**i for i in range(9)]



print(powers_of_two())



assert powers_of_two() == [1, 2, 4, 8, 16, 32, 64, 128, 256]



print('done')
# Question 3



def is_distinct(coll):

     for i in range(0, len(coll) - 1):

         for j in range(i + 1, len(coll)):

             if coll[i] == coll[j]:

                 return False

     return True



assert is_distinct([1, 2])

assert not is_distinct([1, 2, 1])

assert not is_distinct([1, 1])



print('done')
# Question 4



def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h



assert list(harmonic_gen(3)) == [1.0, 1.5, 1.8333333333333333]



print('done')