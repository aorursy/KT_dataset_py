L = [2, 3, 5, 7]

L
# Length of a list

len(L)
# Append a value to the end

L.append(11)



L
# Addition concatenates lists

L + [13, 17, 19]
# sort() method sorts in-place

L = [2, 5, 1, 6, 3, 4]

L.sort()

L
L = [1, 'two', 3.14, [0, 3, 5]]

L[3]
L = [2, 3, 5, 7, 11]
L[0]
L[1]
L[-1]
L[-2]
L[0:3]
L[:-3]
print(L)

L[-3:]
L[::4]  # equivalent to L[0:len(L):2]
L[::-1]
L[0] = 100

print(L)
L[1:3] = [55, 56]

print(L)
t = (1, 2, 3)
t = 1, 2, 3

print(t)
len(t)
t[0]
t[1] = 4
t.append(4)
x = 0.125

x.as_integer_ratio()
numerator, denominator = x.as_integer_ratio()

#print(numerator / denominator)

numerator, denominator 
numbers = {'one':1, 'two':2, 'three':3}

numbers
# Access a value via the key

numbers['two']
# Set a new key:value pair

numbers['ninety'] = 90

print(numbers)
primes = {2, 3, 5, 7}

odds = {1, 3, 5, 7, 8}
# union: items appearing in either

#primes | odds      # with an operator

primes.union(odds) # equivalently with a method
# intersection: items appearing in both

primes & odds             # with an operator

primes.intersection(odds) # equivalently with a method
# difference: items in primes but not in odds

primes - odds           # with an operator

primes.difference(odds) # equivalently with a method
# symmetric difference: items appearing in only onece in each set

primes ^ odds                     # with an operator

primes.symmetric_difference(odds) # equivalently with a method