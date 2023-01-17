list1 = [2, 4, 6, 8]
next(filter(lambda x: x % 2 == 0, list1))
next(filter(lambda x: x % 2 == 0, list1))
is_even = lambda x: x % 2 == 0

is_odd = lambda x: x % 2 == 1
print(next(filter(is_even, list1)))

print(next(filter(is_even, list1)))
try:
    print(next(filter(is_odd, list1)))
except StopIteration:
    print("StopIteration exception is raised.")
print(next(filter(is_odd, list1), 1))