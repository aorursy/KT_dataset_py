x = int(input()) 
if x > 0:
    print(x)
    print("Число положительно")
elif x == 0:
    print(x)
    print("Число равно 0")
else:
    print(-x)
    print("Число отрицательно")
   

x1 = int(input())
x = x1
if x < 0:
    x = -x
print('|' + str(x1) + '|' + ' = ' + str(x))
x = int(input())
y = int(input())
if x > 0:
    if y > 0:               # x > 0, y > 0
        print("Первая четверть")
    else:                   # x > 0, y < 0
        print("Четвертая четверть")
else:
    if y > 0:               # x < 0, y > 0
        print("Вторая четверть")
    else:                   # x < 0, y < 0
        print("Третья четверть")
if not 1>2 and 1<4:
    print(1)
else:
    print(2)
print(int(True), int(False), bool(-1), bool("abcd"), bool(""))
if not bool('') and 2<1:
    print("Пустая строка")
a = int(input())
b = int(input())
if a % 10 == 0 or b % 10 == 0:
    print('YES')
else:
    print('NO')

a = int(input())
b = int(input())
if a > 0 and not b < 0:
    print('YES')
else:
    print('NO')
    
x = int(input())
y = int(input())
if x > 0 and y > 0:
    print("Первая четверть")
elif x < 0 and y > 0:
    print("Вторая четверть")
elif y < 0 and x <0:
    print("Третья четверть")
else:
    print("Четвертая четверть")
a = int(input())
b = int(input())
if a < b:
    print("Минимальное из двух введённых чисел это", a)
else:
    print("Минимальное из двух введённых чисел это", b)
a = -1000000
if a > 0:
    print(1)
elif a == 0:
    print(0)
else:
    print(-1)
x1, y1 = int(input()), int(input())
x2, y2 = int(input()), int(input())
if (x1 + y1) % 2 == (x2 + y2) % 2:
    print('YES')
else:
    print('NO')
year = int(input())
if (year % 4 == 0 and not year % 100 == 0) or year % 400 == 0:
    print('YES')
else:
    print('NO')
x1, y1 = int(input()), int(input())
x2, y2 = int(input()), int(input())
if (x1 == x2) or (y1 == y2):
    print('YES')
else:
    print('NO')
x1, y1 = int(input()), int(input())
x2, y2 = int(input()), int(input())
dx = x2 - x1
dy = y2 - y1
if dx < 0:
    dx = -dx
if dy < 0:
    dy = -dy 
    
if dx == 1 and dy ==2:
    print("YES")
elif dx == 2 and dy ==1:
    print("YES")
else:
    print("NO")
n, m = int(input()), int(input())
x, y = int(input()), int(input())
print(min(x, y, n-y, m-x))

