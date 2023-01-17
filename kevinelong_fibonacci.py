limit = 24

a = 0
b = 1

for n in range(limit):
    c = a + b
    a = b
    b = c
    
#     a, b = b, a + b
    
    print(a, b / a)

