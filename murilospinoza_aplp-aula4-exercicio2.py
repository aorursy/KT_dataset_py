def fibonacci(n):
    if n > 1:
        n = fibonacci(n-1) + fibonacci(n-2)
    return n
fibonacci_l = lambda n: fibonacci_l(n - 1) + fibonacci_l(n - 2) if n > 1 else n
sequencia_funcao = [fibonacci(n) for n in range(0, 21)]
sequencia_lambda = [fibonacci_l(n) for n in range(0, 21)]

print(sequencia_funcao)
print(sequencia_lambda)