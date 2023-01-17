numbers = [1, 2, 2, 2, 3, 4, 5, 6, 7]

numbers
numbers.remove(2)
numbers
# relying on the original list's length

# remove number 2

numbers = [1, 2, 2, 2, 2, 2, 2, 3]



for n in numbers:

    print("List length at start of loop is ", len(numbers))

    numbers.remove(2)

    print("List length at end of loop is", len(numbers))

    print(numbers)



numbers
# relying on the original list's length

# remove number 2

numbers = [1, 2, 2, 2, 2, 2, 2, 2, 3]



# building up an empty list

output = []



for n in numbers:

    # check if two or check if the number is not two...

    # "not equal to"

    if n != 2:

        output.append(n)

        

output        
def remove_twos(numbers):

    output = []



    for n in numbers:

        # check if two or check if the number is not two...

        # "not equal to"

        if n != 2:

            output.append(n)

    return output



# automated test 

assert remove_twos([2, 2, 2, 2, 2, 2, 2, 2, 2]) == []

assert remove_twos([1, 2, 3]) == [1, 3]

print("Remove twos is functioning!")
x = 5

x
x + 10
# IF you need text, use quotation marks.

name = "Stuart"

print(name)
# Recommend using double quotes becausse you might have a contraction

print("We're working on Python together today. Ain't that grand!")
# type is a build-in function

# parentheses mean that you're telling a function to run.

# values that go inside the parentheses are the inputs to that function

print("Hello")
# Functions w/o parens are like a blueprint for a car horn

# Think of functions as VERBS

print
print(34)
# One example of using an indentation

is_raining = False

if is_raining:

    print("I'll bring an umbrella")

print("Have a good day")
is_sunny = True

is_hot = True

if is_sunny and is_hot:

    print("Stay indoors")

    print("Turn on the AC")

    print("Work on Python")
right_here_now = True

on_mars_with_elon = False

if right_here_now and on_mars_with_elon:

    print("We broke logic")
likes_olives = True

likes_anchovies = False

if likes_olives and likes_anchovies:

    print("Get an olive anchovie pizza")
allergic_to_pollen = True

allergic_to_peanuts = False

if allergic_to_pollen or allergic_to_peanuts:

    print("Take an antihystamine")

else:

    print("Have a peanut picnic under a tree")
# assert runs expected vs. actual and throws an error if they're different

# if the assert doesn't do squat, it means that expected value is the same as the actual value

assert 1 == 1

assert 2 == 2
# median function

# use a known good solution

numbers = [1, 5, 2, 3, 4, 6]



# sort the list

# if a list has an odd number of elements, the median is the middle number

# if the list has an even number of elements, then the median is the average of the two middle numbers



# .sort is a function that is accessible from variables that are lists

numbers.sort()



# length of the list is len(numbers)

length = len(numbers)



# % is the remainder operator. It provides the remainder of dividing the first number before % 

# by the number after the %. 



# Remainer "aint" zero, "!=" as "aint"

is_odd = length % 2 != 0

if is_odd:

    index_of_middle = int(len(numbers) / 2)

    median

    print(median)

else:

    # average the two middle values

    first_index = int(len(numbers) / 2)

    second_index = int(first_index - 1)

    total = numbers[first_index] + numbers[second_index]

    median = total / 2

    print(median)
# is_positive, is_positive_even, etc..



number = -12



# > returns True if the value on the left is greater than the number on the right

number > 0
3 > 0
# We need a function (lil' machines) that works for any value



# Parts of a function

# def keyword (means define a function)

# name_of_the_function

# input variable in the parenthesis

# :

# indended lines of code are the "body" of the function

# Always, always use return.



# Function definition (a recipie)

def is_positive(number):

    result = number > 0

    return result



# Function "call", means to run/call/invoke (to follow that recipie)

assert is_positive(5) == True

assert is_positive(-23) == False



assert is_positive(0.00000001) == True
# first_number % second_number returns the remainder

# "%" symbol means "remainder", "modulo"

print(4 % 2)

print(5 % 2)

print(7 % 2)

print(100 % 2)
# Write an "is_even" function that returns True if the input variable is even

# If the input variable has a remainder of 0 after being divided by 2, then reutnr True

def is_even(number):

    remainder = number % 2

    result = remainder == 0 # == returns True if both values are the same

    return result



print(is_even(5))

print(is_even(100))
# Where we are right now:

# we have a function that tells us if a number is positive (True/False)

# we have a function that tells us if a number is even (True/False)

# Our next task is to write a function that checks if a number is BOTH even and positive
# AND

like_olives = True

like_pineapple = True

if like_olives and like_pineapple:

    print("Let's have pineapple and olive pizza")

else:

    print("that sounds awful")
# Build an is_even_and_positive 

number = 12

print(is_even(number))

print(is_positive(number))

print(is_even(number) and is_positive(number))
# Build an is_even_and_positive 

number = 11

print(is_even(number))

print(is_positive(number))

print(is_even(number) and is_positive(number))
# When we run a function with an input, it's as if the result value replaces that line of code.

def is_positive_even(number):

    return is_even(number) and is_positive(number)
False and True
print(is_positive_even(11))

print(is_positive_even(-12))

print(is_positive_even(8))
# Exercise - add 1 to every number in a list

numbers = [3, 4, 5, 6, 7]



# Make an empty list

output = []



for n in numbers:

    # each time through this loop, we add 1 to each number and append to the new list

    output.append(n + 1)

output
# Exercise - add 2 to every number on a list

# list comprehension example

numbers = [7, 8, 9, 10, 11]



# List comprehension is like a one line for loop

# "for n in list_name" - this creates a variable called n

# first time through the loop, n holds the first value of the list

# second time through the loop, n holds the second value of the list

numbers = [n + 2 for n in numbers]

numbers
# Example of a list comprehension

[n * 10 for n in [1, 2, 3]]