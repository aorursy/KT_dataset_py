def fibonacci_generator():
  a = 0
  b = 1
  while True:
    yield a
    a, b = b, a + b
fib_gen = fibonacci_generator()
next(fib_gen)
next(fib_gen)
next(fib_gen)
next(fib_gen)
next(fib_gen)
