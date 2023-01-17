# Create a function that generates a Fibonacci sequence for F0 = 0 and F1 = 1. The function should take one input, n.

# n should be an integer and the function will generate a tuple that contains n numbers where the entire sequence of numbers is a Fibonacci sequence for F0 = 0 and F1 = 1

# If n = 0, return the number zero



## Hint: In addition to range(n), range can also be used as range(start, stop). For example:

###  for i in range(2,5):

###     print(i)

#---

#2

#3

#4

    

# Print out test cases here

print(fibonacci())

print(fibonacci(1))

print(fibonacci(2))

print(fibonacci(10))
# Modify your fibonacci function to generate a sequence given any F0 and F1. F0 and F1 must be different and be a positive integer.



# Print out test cases here

print(fibonacci_mod())

print(fibonacci_mod(1))

print(fibonacci_mod(2))

print(fibonacci_mod(5,0,0))

print(fibonacci_mod(10))

print(fibonacci_mod(10,2,5))

print(fibonacci_mod(5,-2,5))
# Create another function that returns the nth Fibonacci number in a given Fibonacci sequence for the f0, f1. Return 0 if any given inputs are invalid.



# Print out test cases here

print(fibN())

print(fibN(1))

print(fibN(2))

print(fibN(10))

print(fibN(10,2,5))

print(fibN(5,-2,5))