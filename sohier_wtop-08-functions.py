print('abc')
print(1, 2, 3)
print(1, 2, 3, sep='--')
def fibonacci(N):

    L = []

    a, b = 0, 1

    while len(L) < N:

        a, b = b, a + b

        L.append(a)

    return L
fibonacci(10)
def real_imag_conj(val):

    return val.real, val.imag, val.conjugate()



r, i, c = real_imag_conj(3 + 4j)

print(r, i, c)
def fibonacci(N, a=0, b=1):

    L = []

    while len(L) < N:

        a, b = b, a + b

        L.append(a)

    return L
fibonacci(10)
fibonacci(10, 0, 2)
fibonacci(10, b=3, a=1)
def catch_all(*args, **kwargs):

    print("args =", args)

    print("kwargs = ", kwargs)
catch_all(1, 2, 3, a=4, b=5)
catch_all('a', keyword=2)
inputs = (1, 2, 3)

keywords = {'pi': 3.14}



catch_all(*inputs, **keywords)
add = lambda x, y: x + y

add(1, 2)
def add(x, y):

    return x + y
data = [{'first':'Guido', 'last':'Van Rossum', 'YOB':1956},

        {'first':'Grace', 'last':'Hopper',     'YOB':1906},

        {'first':'Alan',  'last':'Turing',     'YOB':1912}]
sorted([2,4,3,5,1,6])
# sort alphabetically by first name

sorted(data, key=lambda item: item['first'])
# sort by year of birth

sorted(data, key=lambda item: item['YOB'])