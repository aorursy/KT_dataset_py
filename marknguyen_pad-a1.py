# Import fib1.csv file and save all row data as integers

# Use this code to import data from a Kaggle kernel. Make sure you import the dataset for this assignment first into your Kaggle kernel



with open('../input/fib1.csv','r') as fin:

    seq = [int(r) for r in fin.readlines()[1:]] # Build list using list comprehension: https://medium.com/better-programming/list-comprehension-in-python-8895a785550b



print(seq)
# Confirm whether the numbers contained in fib1.csv is a Fibonacci sequence with F0 = 0 and F1 = 1

# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient

# Confirm whether the numbers contained in fib2.csv is a Fibonacci sequence with F0 = 4 and F1 = 8

# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient

# Confirm whether the numbers contained in fib3.csv is a Fibonacci sequence with F0 = 2 and F1 = 6

# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient
