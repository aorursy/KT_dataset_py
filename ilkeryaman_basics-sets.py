x = set()
print(type(x))
x = {1, 2, 3}
print(x)
list1 = [1, 2, 3, 4, 1, 2, 3]
x = set(list1)

print(x)
x.add(5)

print(x)
y = set([1, 4, 8, 9])
print(x)

print(y)

print(x.difference(y))

print(y.difference(x))

print(x.intersection(y))

print(x.union(y))
x.discard(3)

print(x)