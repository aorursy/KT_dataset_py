list_1 = [1, 2, 3]

for i, value in enumerate(list_1):

    print(i, ':', value)
def func():

    return (1, 2, 3)
a, b, c = func()

print(a, b, c)
import sys

x = 1, 2, 3

print(sys.getsizeof(x))
list_2 = [1, 5, 4, 3, 4, 5, 3, 2, 5, 1, 2, 4, 5, 9, 6 ,7, 4]

print(max(set(list_2), key=list_2.count))
func = lambda a: a + 10

func(5)
n = 10

res = 1 < n < 20

print(res)

res = 1 > n <= 20

print(res)
alphabets = []



for i in range(97, 123):

    alphabets.append(chr(i))



print(alphabets)