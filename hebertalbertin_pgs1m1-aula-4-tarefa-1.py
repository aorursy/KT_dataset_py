def f(i):
    if i == 1:
        i = 2
    elif i == 2:
        i = 1
    else:
        i = 2*f(i-1) + g(i-2)
    return i        
def g(i):
    if i >= 3:
        i = g(i-1) + 3*f(i-2)
    return i
def k(n):
    if(n > 0):
        return (f(n), g(n))
print(k(2))
print(k(3))
print(k(4))
# List Comprehension
[k(i) for i in range(1, 6)]
# lambda
fl = lambda x: 1 if x == 2 else 2 if x == 1 else 2*fl(x-1) + gl(x-2)
gl = lambda x: gl(x-1) + 3*fl(x-2) if x >= 3 else x
kl = lambda x: None if x < 0 else (f(n), g(n))
[k(i) for i in range(1, 6)]
