print("Good afternoon, ya'll!")
# assigning a value to a variable

today = "Today is a great day to learn"

print(today)
print("Use the print command")

print("To print")

print("Prints each and every line")
"Notebook cells automatically print the last line of code"
# run this cell and the number 5 will be printed

5
"Multiple lines"

"do not print automatically"

"only the last line"
"I'm writing Python with ya'll right now"
print("That means")

print("The print command is important.")

print("When we want to print each line")
# Exercise: Print your name in this cell. Be sure to use quotation marks.

# Exercise: Print your favorite number in this cell

# Exercise: Print the number 7 in this cell

# Exercise: Print out 3 + 4 in this cell

# Make Python do the math rather than adding 3 + 4 in your head

# Exercise: Print out 3 * 5 in this cell, * means multiply

# This is a comment. Run this cell. Do you see anything? Why or why not?
# First, run this cell.

# Then delete the hashtag at the beginning of the last line.

# Last, run the cell again.

# print("This is a 'commented out' print command")
# In Python

# Comments are a best-practice when it comes to communicating to your future self and to fellow developers

# print("Once the # at the beginning of this line is removed, this string will print.") # additional hashtags make additional comments
# Variables hold values that we use later in the program by name.

message = "Hello, Everybody!"

print(message)
favorite_quote = "If you can't solve a problem, then there is an easier problem you can solve:  find it."

print(favorite_quote)
# Variables are assigned to point to values.

favorite_food = "lasagna"

print("My favorite food is", favorite_food)
# Variables can also be re-assigned to point to different values.

favorite_food = "pizza" 

print("I changed my mind. My favorite food is actually", favorite_food)
# Exercise: Reassign the favorite_food variable to hold your own favorite food, then print the variable.

# Exercise: Create and assign the variable favorite_number to hold your own favorite number.

# Exercise: Reassign the favorite_food variable to hold the number 7. 

# What happens and why? Does this match your expectations?

# The "None" data type represents the absence of a value. This is called "null" in some other programming languages

type(None)
type(True)
type(False)
# Numbers without a decimal are integers. Integers can be negative or positive.

type(-99)
type(2.3)
.5 * .5
type("Howdy!")
# True and False are the only two Boolean values.

True

False



print(True)

print(False)
# Exercise: Add 1 plus 2 plus 3 plus 4 plus 5 plus 6 plus 7 plus 8 plus 9 plus 10

# Exercise: Subtract 23 from 55

# Exercise: Multiply 2.3 by 4.4. What's the data-type that this returns?

# Exercise: Multiply 3 by 5.

# Exercise: Print out the data-type of the result of multiplying 11 * 7

# Exercise: Divide 1 by 2. What happens? What data type is returned?

# Exercise: Divide 5 by 0. What happens? Why do you think that is?

# Exercise: Use the modulo operator to obtain the remainder of dividing 5 by 2

# Exercise: Use the modulo operator to obtain the remainder of dividing 8 by 2

# Exercise: Use the modulo operator to obtain the remainder of dividing 9 by 3

# Exercise: Use the modulo operator to obtain the remainder of dividing 7 by 2

# Exercise: Use the exponent operator to raise 2 to the 3rd power

# Exercise: Use the exponent operator to raise 10 to the 10th power

# Exercise: Use the exponent operator to raise 100 to the 100th power

# Exercise: Use the exponent operator to raise 2 to the negative 1st power

# Exercise: Use the exponent operator to raise 123 to the 0th power

# Exercise: Use what you have learned to determine the average of the following variables

a = 5

b = 7

c = 9

d = 17

e = 11



# "Not" operator changes the boolean value. (Only booleans are True and False)

is_raining = False

not is_raining
# Boolean values assigned to variables 

right_here_now = True

learning_python = True

on_the_moon = False
print(right_here_now and on_the_moon)
# Read the line of code below and think about the result. 



print(learning_python or on_the_moon)
indenting = False

if indenting:

    print("We are indenting b/c the indent means the indended stuff belongs to the if")

print("This happens no matter what")
if right_here_now:

    print("So glad you made it today!")

    print("Wherever you go, there you are.")

else:

    print("You are not right here now.")

    

print("Unindented means the if/else is over, and we're back to our regularly scheduled programming.")
# Before running this cell, think about and predict the outcome.

if on_the_moon:

    print("You are currently on a lunar base.")

else:

    print("You are not on the moon!")
# Before running this cell, think about and predict the outcome.

if right_here_now and learning_python:

    print("We are right here now and learning python.")
# Before running this cell, think about and predict the outcome.

if right_here_now and learning_python and on_the_moon:

    print("You must be magic since you're on the moon and here and learning Python all at the same time.")
# Before running this cell, think about and predict the outcome.

if right_here_now or on_the_moon:

    print("How many True values does it take with an OR for the entirety to be True?")
# Before running this cell, think about and predict the outcome.

# Carefully consider the parentheses...

if learning_python and (right_here_now or on_the_moon):

    print("In addition to learning python, you are either right here now or on the moon")
# Exercise: Create a variable called is_raining and assign it True or False depending on if it's raining right now

# Exercise: Create a variable named will_rain_later and assign a value of True or False if it may rain later today

# Exercise: Create an "if" condition that checks if it's raining now or later. 

# If it is going to rain now or later, then print out "I'll bring an umbrella"

print(5 == 5)

print(-4 + 4 == 0)
5 < 10
print("Hello" == "Goodbye")

print("Hello" != "Goodbye") # != means "not equal to" 

# = assignment

# == comparison

# != not the same
# Lists are created with square brackets. Each item on the list is separated by a comma. Lists can hold any data type

beatles = ["John", "Paul", "George"]

print(beatles)
# type out the variable, add a period, then hit TAB

print("How many times John shows up is", beatles.count("John"))

beatles.reverse()

beatles
# Lists have built-in functions

beatles.append("Ringo") # Don't forget Ringo

print(beatles)
beatles
# .append and .reverse are some of the built-in list functions

beatles.reverse()

print(beatles)
beatles.append("all of us")

beatles
# We can also use lists to hold numbers

odds = [1, 3, 5, 7, 9, 11, 13, 17, 19, 21]
# for variable in list:

# For each number in the list of odds, do something

# thing we do each time through the loop IS the code indented within that loop

for number in odds:

    print(number)

    print("The loop is going")

print("The Loop is complete.")
for number in odds:

    print(number + 10)
odds
# sum is a built in function

numbers = [2, 3, 5, 7]

sum(numbers)
# len is a built-in function short for length

len(numbers)
sum(numbers) / len(numbers)
# The len function also works on strings

programming_language = "Python"

print("The number of letters in Python is", len(programming_language))
# how to define our own functions

# def function_name(variable_name_for_the_input):

#      output = guts of the function

#      return output

def is_even(number):

    remainder = number % 2 # the % operator returns the remainder of integer division of the number on the left by the number on the right

    if remainder == 0: # even numbers have no remainder

        return True

    else:

        return False



# calling a function means to run it

print(is_even(2))

print(is_even(4))

print(is_even(2.3))

print(is_even(3))
# lots of built in functions work on sequences

names = ["John", "Paul", "George", "Ringo"]



print("The number of items on the list is", len(names))

numbers = [2, 3, 5, 7]

print("The highest number on the list is", max(numbers))

print("The smallest number on the list is", min(numbers))
# def keyword to define, name_of_function, parentheses for parameters

def square(number):

    return number * number # return is the output we hand back to the code that called the function (calling == execute == run)
square(3)
square(4)
# How to average a list of numbers.

numbers = [2, 3, 5, 7]

total = sum(numbers)

average = total / len(numbers)
