multiply = lambda a, b: a * b
multiply(2, 3)
try:
    print(multiply(3, "y"))
except ValueError:
    print("A ValueError is thrown.")
finally:
    print("Finally block is executed.")
multiply = lambda a, b: a * int(b)
try:
    print(multiply(3, "y"))
except ValueError:
    print("A ValueError is thrown.")
finally:
    print("Finally block is executed.")