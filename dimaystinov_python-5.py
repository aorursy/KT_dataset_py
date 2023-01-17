s = input()
print(len(s))
t = input()
number = int(t)
u = str(number)
print(s * 3)
print(s + ' ' + u)
s = 'abcdefg'
print(s[1])
print(s[-1])
print(s[1:3])
print(s[1:-1])
print(s[:3])
print(s[2:])
print(s[:-1])
print(s[::2])
print(s[1::2])
print(s[::-1])
S = 'Hello'
print(S.find('e'))
# вернёт 1
print(S.find('ll'))
# вернёт 2
print(S.find('L'))
# вернёт -1
print(S.rfind('l'))
# вернёт 2
print(S.rfind('l'))
# вернёт 3
print('Hello'.replace('l', 'L'))
# вернёт 'HeLLo'
print('Abrakadabra'.replace('a', 'A', 2))
# вернёт 'AbrAkAdabra'
print('Abracadabra'.count('a'))
# вернёт 4
print(('a' * 10).count('aa'))
# вернёт 5
