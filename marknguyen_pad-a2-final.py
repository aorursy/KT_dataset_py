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

    

def fibonacci(n = 0):

    """Basic Fibonacci sequence generator. Returns a list of numbers that is a Fibonacci sequence for F0 = 0 and F1 = 1

    

    Args:

        n (int): Size of returned sequence

        

    Returns:

        list: Fibonacci sequence with size n 

    """

    

    f0 = 0

    f1 = 1

    

    # Guard statements for n

    if n == 0:

        return 0

    elif n == 1:

        return [f0]

    elif n < 0:

        print("Invalid input, please enter a positive integer")

        return 0

    

    seq = [f0,f1]

    for i in range(2,n): # n should always be > 1 by this point in the code

        seq.append(seq[i-2] + seq[i-1])

    

    return seq
# Print out test cases here

print(fibonacci())

print(fibonacci(1))

print(fibonacci(2))

print(fibonacci(10))
# Modify your fibonacci function to generate a sequence given any F0 and F1. F0 and F1 must be different and be a positive integer.



def fibonacci_mod(n = 0, f0 = 0, f1 = 1):

    """Basic Fibonacci sequence generator. Returns a list of numbers that is a Fibonacci sequence for F0 = 0 and F1 = 1.

    F0 and F1 must not be both 0

    

    Args:

        n (int): Size of returned sequence

        f0 (int): Value must be >= 0

        f1 (int): Value must be >= 0

        

    Returns:

        list: Fibonacci sequence with size n 

    """

    

    # Add additional guard statements for f0 and f1

    if f0 < 0 or f1 < 0:

        print("f0 and f1 are invalid. They both must be positive integers.")

        return 0

    elif f0 + f1 == 0:

        print("f0 and f1 cannot both be zero values.")

        return 0

    

    # Guard statements for n

    if n == 0:

        return 0

    elif n == 1:

        return [f0]

    elif n < 0:

        print("Invalid input, please enter a positive integer")

        return 0

    

    seq = [f0,f1]

    for i in range(2,n): # n should always be > 1 by this point in the code

        seq.append(seq[i-2] + seq[i-1])

    

    return seq
# Print out test cases here

print(fibonacci_mod())

print(fibonacci_mod(1))

print(fibonacci_mod(2))

print(fibonacci_mod(5,0,0))

print(fibonacci_mod(10))

print(fibonacci_mod(10,2,5))

print(fibonacci_mod(5,-2,5))
# Create another function that returns the nth Fibonacci number in a given Fibonacci sequence for the f0, f1. Return 0 if any given inputs are invalid.



def fibN(n = 0, f0 = 0, f1 = 1):

    """Return the nth number of a Fibonacci sequence defined by f0 and f1

    

    Args:

        n (int): The nth number of the Fibonacci sequence

        f0 (int): Value must be >= 0

        f1 (int): Value must be >= 0

        

    Returns:

        int: nth number of Fibonacci sequence

    """

    

    seq = fibonacci_mod(n,f0,f1)

    

    if seq == 0:

        return 0

    else:

        return seq[-1]
# Print out test cases here

print(fibN())

print(fibN(1))

print(fibN(2))

print(fibN(10))

print(fibN(10,2,5))

print(fibN(5,-2,5))