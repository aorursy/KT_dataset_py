# Question 1



# 2^10                  O(1)

# 2 long(n)             O(log(n))

# 3n + 100log(n)        O(n)

# 4n                    O(n)

# nlog(n)               O(n log(n))

# 4n log(n) + 2n        O(n Log(n))

# n^2 + 10n             O(n^2)

# n^3                   O(n^3)

# 2^n                   O(2^n)





# Question 2



# Let 8n log(n) = 2n^2

# Therefore, 4log(n) = n

# 4 log(16) = 4 * 4

#    = 16

# Therefore A is faster when n >= 16





# Question 3



# d(n) = O(f(n)),





# Question 4



# 2 operations outside of the loop, then 3 operations inside the loop which will each happen

# on each iteration of the loop determined by the length on s, 'n'.

# Runtime O(n) = 3n + 2



def example1(s):

    """Return the sum of the elements in sequence S."""

    n = len(s)

    total = 0

    for j in range(n):  # loop from 0 to n-1

        total += s[j]

    return total





# Question 5



# 2 operations outside of the loop, 3 operations inside the loop.

# Loop only iterates for half the length of n due to the increment of 2

# in the loop condition.

# O(n) = 3n/2 + 2





def example2(s):

    """Return the sum of the elements with even index in sequence S."""

    n = len(s)

    total = 0

    for j in range(0, n, 2):  # note the increment of 2

        total += s[j]

    return total





# Question 6



# 2 operations outside of the loop, inner loop iterating 1+n times with 3 operations

# O(n^2) = 1.5n^2 + 1.5n + 2





def example3(s):

    """Return the sum of the prefix sums of sequence S."""

    n = len(s)

    total = 0

    for j in range(n):  # loop from 0 to n-1

        for k in range(1 + j):  # loop from 0 to j

            total += s[k]

        return total





# Question 7



# 3 operations outside of the loop, 5 operations on each iteration.

# O(n) = 5n + 3





def example4(S):

    """Return the sum of the prefix sums of sequence S."""

    n = len(S)

    prefix = 0

    total = 0

    for j in range(n):

        prefix += S[j]

        total += prefix

        return total





# Question 8



# O(n^3) = 2 + n(1 + 1.5n(n+1) + 5)





def example5(a, b):  # assume that A and B have equal length

    """Return the number of elements in B equal to the sum of prefix sums in A."""

    n = len(a)

    count = 0

    for i in range(n):  # loop from 0 to n-1

        total = 0

        for j in range(n):  # loop from 0 to n-1

            for k in range(1 + j):  # loop from 0 to j

                total += a[k]

                if b[i] == total:

                    count += 1

                    return count
