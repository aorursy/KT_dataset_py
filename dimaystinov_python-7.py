Primes = [2, 3, 5, 7, 11, 13]
Rainbow = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
Rainbow = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
print(Rainbow[0])
Rainbow[0] = 'красный'
print('Выведем радугу')
for i in range(len(Rainbow)):
    print(Rainbow[i])
5
1809
1854
1860
1891
1925
a = []  # заводим пустой список
n = int(input())  # считываем количество элемент в списке
for i in range(n):  
    new_element = int(input())  # считываем очередной элемент
    a.append(new_element)  # добавляем его в список
    # последние две строки можно было заменить одной:
    # a.append(int(input()))
print(a)
a = []
for i in range(int(input())):
    a.append(int(input()))
print(a)
a = [1, 2, 3]
b = [4, 5]
c = a + b
d = b * 3
print([7, 8] + [9])
print([0, 1] * 3)
a = [0] * int(input())
for i in range(len(a)):
    a[i] = int(input())
a = [1, 2, 3, 4, 5]
for i in range(len(a)):
    print(a[i])
a = [1, 2, 3, 4, 5]
for elem in a:
    print(elem, end=' ')

# дано: s = 'ab12c59p7dq'
# надо: извлечь цифры в список digits,
# чтобы стало так:
# digits == [1, 2, 5, 9, 7]

s = 'ab12c59p7dq'
digits = []
for symbol in s:
    if '1234567890'.find(symbol) != -1:
        digits.append(int(symbol))
print(digits)
# на вход подаётся строка
# 1 2 3
s = input()  # s == '1 2 3'
a = s.split()  # a == ['1', '2', '3']
a = input().split()
for i in range(len(a)):
    a[i] = int(a[i])
a = [int(s) for s in input().split()]
a = '192.168.0.1'.split('.')

a = ['red', 'green', 'blue']
print(' '.join(a))
# вернёт red green blue
print(''.join(a))
# вернёт redgreenblue
print('***'.join(a))
# вернёт red***green***blue

a = [1, 2, 3]
print(' '.join([str(i) for i in a]))
# следующая строка, к сожалению, вызывает ошибку:
# print(' '.join(a))
n = 5
a = [0] * n
[выражение for переменная in последовательность]
a = [0 for i in range(5)]
n = 5
a = [i ** 2 for i in range(n)]
n = 5
a = [i ** 2 for i in range(1, n + 1)]
from random import randrange
n = 10
a = [randrange(1, 10) for i in range(n)]
a = [input() for i in range(int(input()))]
A = [1, 2, 3, 4, 5]
A[2:4] = [7, 8, 9]
A = [1, 2, 3, 4, 5, 6,  7]
A[::-2] = [10, 20, 30, 40]