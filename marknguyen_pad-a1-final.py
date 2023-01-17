# Import fib1.csv file and save all row data as integers

# Use this code to import data from a Kaggle kernel. Make sure you import the dataset for this assignment first into your Kaggle kernel



with open('../input/fib1.csv','r') as fin:

    seq = [int(r) for r in fin.readlines()[1:]] # Build list using list comprehension: https://medium.com/better-programming/list-comprehension-in-python-8895a785550b



print(seq)
# Confirm whether the numbers contained in fib1.csv is a Fibonacci sequence with F0 = 0 and F1 = 1

# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient



if seq[0] != 0 or seq[1] != 1:

    print("Sequence does not have F0 = 0 and F1 = 1")

    

## Build our own Fibonnaci sequence with F0 = 0 and F1 = 1

seq_built = [0,1,1,2,3,5,8,13,21,34]



## Check each sequence



f2 = seq_built[2] == seq[2]

f3 = seq_built[3] == seq[3]

f4 = seq_built[4] == seq[4]

f5 = seq_built[5] == seq[5]

f6 = seq_built[6] == seq[6]

f7 = seq_built[7] == seq[7]

f8 = seq_built[8] == seq[8]

f9 = seq_built[9] == seq[9]



total_correct = f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 # True booleans get converted to 1 and False booleans get converted to 0 when adding booleans together



if (total_correct == 8):

    print("fib1.csv contains a Fibonacci sequence with F0 = 0 and F1 = 1")

else:

    print("fib1.csv does not contain a Fibonacci sequence with F0 = 0 and F1 = 1")
# Confirm whether the numbers contained in fib2.csv is a Fibonacci sequence with F0 = 4 and F1 = 8

# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient

with open('../input/fib2.csv','r') as fin:

    seq = [int(r) for r in fin.readlines()[1:]]



if seq[0] != 4 or seq[1] != 8:

    print("Sequence does not have F0 = 4 and F1 = 8")

    

## Build our own Fibonnaci sequence with F0 = 4 and F1 = 8

seq_built = [4,8,12,20,32,52,84,136,220,356]



## Check each sequence



f2 = seq_built[2] == seq[2]

f3 = seq_built[3] == seq[3]

f4 = seq_built[4] == seq[4]

f5 = seq_built[5] == seq[5]

f6 = seq_built[6] == seq[6]

f7 = seq_built[7] == seq[7]

f8 = seq_built[8] == seq[8]

f9 = seq_built[9] == seq[9]



total_correct = f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 # True booleans get converted to 1 and False booleans get converted to 0 when adding booleans together



if (total_correct == 8):

    print("fib2.csv contains a Fibonacci sequence with F0 = 4 and F1 = 8")

else:

    print("fib2.csv does not contain a Fibonacci sequence with F0 = 4 and F1 = 8")
# Confirm whether the numbers contained in fib3.csv is a Fibonacci sequence with F0 = 2 and F1 = 6

# If you're familiar with loops from another programming language, feel free to research how you would use loops in Python to make your code more efficient

with open('../input/fib3.csv','r') as fin:

    seq = [int(r) for r in fin.readlines()[1:]]



if seq[0] != 2 or seq[1] != 6:

    print("Sequence does not have F0 = 2 and F1 = 6")

    

## Build our own Fibonnaci sequence with F0 = 2 and F1 = 6

seq_built = [2,6,8,14,22,36,58,94,152,246]



## Check each sequence



f2 = seq_built[2] == seq[2]

f3 = seq_built[3] == seq[3]

f4 = seq_built[4] == seq[4]

f5 = seq_built[5] == seq[5]

f6 = seq_built[6] == seq[6]

f7 = seq_built[7] == seq[7]

f8 = seq_built[8] == seq[8]

f9 = seq_built[9] == seq[9]



total_correct = f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 # True booleans get converted to 1 and False booleans get converted to 0 when adding booleans together



if (total_correct == 8):

    print("fib3.csv contains a Fibonacci sequence with F0 = 4 and F1 = 8")

else:

    print("fib3.csv does not contain a Fibonacci sequence with F0 = 4 and F1 = 8")