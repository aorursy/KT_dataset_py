"""

(R-1.1) Write a short Python function, ​is_multiple​(​n, m​),

that takes two integer values and returnsTrue if n is a multiple of m,

that is, ​n​ = ​mi​ for some integer i, and False otherwise

"""



def is_multiple(n,m):



    return True if n % m == 0 else False



print(is_multiple(50,5))

print(is_multiple(60,7))
"""

(R-1.11) Demonstrate how to use Python’s list comprehension syntax to

produce the list [1, 2, 4, 8,16, 32, 64, 128, 256].



The pow() function returns the value of x to the power of y (xy).

If a third parameter is present, it returns x to the power of y, modulus z.

"""



x = [pow(2, k) for k in range (0,9,1)]

print(x)
"""

(C-1.15) Write a Python function that takes a sequence of numbers

and determines if all the numbers are different from each other (that is, they are distinct).

"""



def numUnique(data):

    list = set()

    return not any(i in list or list.add(i) for i in data)



print(numUnique([1,2,3,4,5]))

print(numUnique([1,1,1,1]))
"""

The n-th harmonic number is the sum of the reciprocals of the first n natural numbers.

For example,H​3​ = 1 + ½ + ⅓ = 1.833. A simple Python function for creating a list of the

first n harmonic numbers follows

:def harmonic_list(n):result = []h = 0for i in range(1, n + 1):h += 1 / iresult.append(h)return

resultConvert this function into a generator ​harmonic_gen(n)​ that ​yields​each harmonic number.

"""





def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

    yield h