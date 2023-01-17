def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield b
        a, b = b, a + b
list(fib(6))
g=fib(10)
print(g)
print(next(g), next(g), next(g))
print(next(fib(10)), next(fib(10)), next(fib(10)))
list(fib(-10))
g=fib(1)
next(g)
next(g)