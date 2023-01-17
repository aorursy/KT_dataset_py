def f(n):
    if n == 1:
        n = 2
    elif n == 2:
        n = 1
    else:
        n = 2 * f(n-1) + g(n-2)
    return n

def g(n):
    if n >= 3:
        n = g(n-1) + 3 * f(n-2)
    return n
def k(n):
    if n > 0:
        return (f(n), g(n))
k(2)
k(3)
k(4)
[k(i) for i in range(1, 6)]
f = lambda n: 1 if n == 2 else 2 if n == 1 else 2 * f(n-1) + g(n-2)
g = lambda n: g(n-1) + 3 * f(n-2) if n >= 3 else n
k = lambda n: (f(n), g(n))
[k(i) for i in range(1, 6)]