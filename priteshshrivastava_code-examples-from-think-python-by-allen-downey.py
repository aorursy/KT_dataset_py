print ('Hello World!')
84 * -2
6 ** 2
6 ^ 2
type('1,00,000')
x = y = 1
print(x)
print(y)
z=3,
print(z)
(4/3) * 3 * x**3
minutes = 105
hours = minutes / 60
print (hours)
hours = minutes // 60
print(hours)
rem = minutes % 60
print(rem)
5 != 6
2 and True
x = 5
if x%2 == 0 :
    print("even")
else :
    print("odd")
def countdown(n):
    if n <= 0:
        print('Blastoff!')
    else:
        print(n)
        countdown(n-1)
countdown(5)
def print_n(s, n):
    if n <= 0:
        return
    print(s)
    print_n(s, n-1)
print_n("hello world", 5)
#### Infinite Recursion
def recurse():
    recurse()
recurse()
