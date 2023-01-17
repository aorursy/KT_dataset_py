# Question 1

def isMultiple(n, m):

    return m % n == 0;

    

print(isMultiple(5, 25))



# Question 2

def powerOfTwo():

    twoList = [2**i for i in range(9)]

    print(twoList)



powerOfTwo()



# Question 3

def distinctChecker(numList):

    counter = 0

    # for each number

    for i in range(0, len(numList) - 1):

        for j in range(i + 1, len(numList)):  

            if str(numList[i]) == str(numList[j]):

                return False

    

    return True

    

        

print(distinctChecker([3, 1, 670, 4, 8, 9, 670]))



# Question 4

def harmonic_gen(n):

        

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h



new_list = []

for value in harmonic_gen(10):

    new_list.append(value)

    

print(new_list)