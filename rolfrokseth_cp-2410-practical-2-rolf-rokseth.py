S = [1,2,3,4,5,90,12,3,123,134,34,15]



def example1(S):

    """Return the sum of the elements in sequence S.""" 

    """2 operations before the loop)""" 

    n = len(S)

    total = 0 

    """ 3 operations for every time the loop runs"""

    for j in range(n):

        total += S[j] 

    return total

example1(S)

#Runtime is 2 + 3n = O(n)

#sum(S)
S = [1,2,3,4,5,90,12,3,123,134,34,15]



def example2(S):

    """Return the sum of the elements with even index in sequence S."""

    n = len(S)   

    total = 0

    """2 operations before the loop"""

    for j in range(0, n, 2):

        total += S[j]

    """Loop run n/2, which gives 1.5n. """

    return total

example2(S)

#2 + 1.5n = O(n)
S = [1,2,3,4,5,90,12,3,123,134,34,15]





def example3(S):

    """Return the sum of the prefix sums of sequence S."""

    n = len(S)   

    total = 0

    """2 operations before the loop"""

    for j in range(n): 

        for k in range(1 + j):

            total += S[k]

    return total



example3(S)

#"""Overall runtime is 2+1.5n(n + 1) = 2 + 1.5n2 + 1.5n = O(n^2)"""
S = [1,2,3,4,5,90,12,3,123,134,34,15]





def example4(S):

    """Return the sum of the prefix sums of sequence S."""

    n = len(S)   

    prefix = 0

    total = 0

    """3 operations before the loop"""

    for j in range(n):       

        prefix += S[j]       

        total += prefix

    return total

"""5 operations in the loop"""



example4(S)



#3 + 5n =O(n)
A = [1,2,3,5,6]

B = [3,6,6,4,4]





def example5(A, B):

    """Return the number of elements in B equal to the sum of prefix sums in A."""

    n = len(A)   

    count = 0

    """2 operatiors before loop"""

    for i in range(n): 

        total = 0

        for j in range(n):  

            for k in range(1 + j): 

                    total += A[k]

            if B[i] == total:           

                count += 1

    return count

    

example5(A,B)