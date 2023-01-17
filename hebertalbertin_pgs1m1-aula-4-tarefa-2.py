def f(n):
    if n > 1:
        n = f(n-1) + f(n-2)
    return n
# lambda
fl = lambda x: fl(x-1) + fl(x-2) if x > 1 else x
print({i: f(i) for i in range(0, 21)})
print('\nlambda:')
print({i: fl(i) for i in range(0, 21)})
