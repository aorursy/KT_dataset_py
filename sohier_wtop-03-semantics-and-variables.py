x = 1         # x is an integer

x = 'hello'   # now x is a string

x = [1, 2, 3] # now x is a list
x = [1, 2, 3]

y = x
print(y)
x.append(4) # append 4 to the list pointed to by x

print(y) # y's list is modified as well!
x = 'something else'

print(y)  # y is unchanged
x = 10

y = x

x += 5  # add 5 to x's value, and assign it to x

print("x =", x)

print("y =", y)
x = 4

type(x)
x = 'hello'

type(x)
x = 3.14159

type(x)
L = [1, 2, 3]

L.append(100)

print(L)
x = 4.5

print(x.real, "+", x.imag, 'i')
x = 4.5

x.is_integer()
x = 4.0

x.is_integer()
type(x.is_integer)