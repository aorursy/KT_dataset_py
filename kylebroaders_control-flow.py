a = 1
if a is not 1:                   # Evaluates "a is not 1" and executes the enclosed text only if it is True
                                 # I could have used != instead of 'is not' and meant the same thing.
    print("It's not the one.")
else:                            # This part runs only if all the rest are False
    print("I found the one!")
classColors = ["blue","yellow","red","green","purple"] # This is a list of strings
myColor = input("What is your favorite color? ")       # The input function asks for input from the user

if myColor in classColors:               # Checks if your input value is present in the classColors list
    print("That's a MHC class color!")
elif myColor is "":                      # Checks if the input is an empty string
    print("You're no fun.")
else:                                    # Default answer if none of the others are true
    print("That's a good color.")
# Exercise 1

input_num = 3

# YOUR CODE HERE
if input_num is 0:
    print("This is nothing")
elif input_num < 0:
    print("This is less than nothing")
elif input_num % 2 is 1:
    print("This is odd")
else:
    print("This is fine")
for letter in "MHC":                 # "letter" is a variable I'm declaring here to hold a different value for each trip through the loop
    print("Give me a "+ letter +"!") # Using string concatenation to each letter
print("Go Lyons!")
print("Printing range(5)")
for i in range(5):
    print(i)
print("Printing range(3,5)")
for i in range(3,5):
    print(i)
print("Printing range(1,10,2)")
for i in range(1,10,2):
    print(i)
# Exercise 2
prime_list = [1,2] # Here's a start to your list

# YOUR CODE HERE
for n in range(3,1001,2):        # Start at 3 and only do odd numbers
    found_prime = True           # Assume the number is prime
    for prime in prime_list[2:]: # Go through the prime list and check to see if the number is evenly divisible by any
        if n % prime is 0:
            found_prime = False
    if found_prime:
        prime_list.append(n)

print(prime_list)
# Here's break in use
for i in range(100):
    if i > 5:
        break
    print(i)
# Here's continue in use
for i in range(10):
    if i % 2 is 0:
        continue
    print(i)
a=1
loopcount=0
while a<10:
    print (a)
    a = a * 1.1
    loopcount = loopcount + 1
print("It took "+ str(loopcount)+" loops to go past 10")
fiblist  = [0,1]
next_fib = 1                           # make a value to store the next number in the sequence

# YOUR CODE HERE
while next_fib<100000:                 # run until the next value is over 100000
    fiblist.append(next_fib)           # append the next value
    next_fib = fiblist[-2]+fiblist[-1] # calculate the next value as the sum of the last two numbers in the sequence

print(fiblist)
# Another version
fiblist  = [0,1]

# YOUR CODE HERE
while fiblist[-1]<100000:                        # run until the last value is over 100000
    fiblist.append(fiblist[-2]+fiblist[-1])      # append the sum of the last two values

fiblist = fiblist[:-1] # The last value goes over 100000, so this line removes the last value

print(fiblist)
