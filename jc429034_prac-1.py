def is_multiple(n, m):

    if n % m == 0:

        result = True

    else:

        result = False



    return result





numbers = [2 ** i for i in range(0, 9)]
def num_check(numList):

    for i in range(0, len(numList) - 1):

        for j in range(i + 1, len(numList)):

            if numList[i] == numList[j]:

                return False

    return True



def harmonic_gen(n):



    h = 0

    for i in range(1, n + 1):

        h += 1 / i



    yield h


