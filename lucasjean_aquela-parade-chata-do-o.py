import math 

a = float(input())

b = float(input())

c = float(input())

d = float(input())

e = float(input())

x = (a + b + c + d + e) / 5

y = (a - x) + (b - x) + (c - x) + (d - x) + (e - x)

w = (y * y) / 4

f = math.sqrt(w)

print(f)
