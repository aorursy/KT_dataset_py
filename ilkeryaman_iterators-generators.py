def get_square():
    for x in range(1, 5):
        yield x ** 2
iterator = iter(get_square())
try:
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
except StopIteration:
    print("Iteration is stopped!")
iterator = iter(get_square())

print("Start of first iterator loop:")
for i in iterator:
    print(i)

print("Start of second iterator loop:")
for i in iterator:
    print(i)         # it does not iterate anymore, because there is no other value.
def multiplication_table():
    for i in range(1, 11):
        print("""
======
 {}'s
======""".format(i))
        for j in range(1, 11):
            yield "{} x {} = {}".format(i, j, i*j)
iterator = iter(multiplication_table())

for x in iterator:
    print(x)