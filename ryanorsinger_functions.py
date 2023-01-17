quote = "Programming languages allow us to formalize instructions and express logic, business rules, mathematics, processes, and automation instructions in one single language in a way where computers follow those instructions to create utility for people"
sum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

sum(range(1, 10)) # range(start, end) produces a range including the start, excluding the end



print(sum(range(1, 11)))
# f(x) = x + 2

# define a function w/ "def" followed by the name of the function

# parameters go in parenthesis, colon after the parens

# body of the function is indented 

# return a value

def add_two(x):

    result = x + 2

    return result



print(add_two(3)) # calling the function is where we "send in" a value

print(add_two(5)) 

# Write a function called identity that takes in a variable and returns that variable.

# Write a function named times_two_plus_2 that takes in a number 

# Write a function named times_three that takes in a number and returns that number times 3
# Write a function named is_even and returns True or False if that number is evenly divisible by 2

def is_even(x):

    result = x % 2 == 0 # is the remainder of dividing that number by 2 zero or not?

    return result



print(is_even(4))

print(is_even(5))

# Write a function named is_odd and returns True or False if that number is odd

# HINT 1: use a search engine to research the answer to this problem

# HINT 2: if your is_even function is working, what about returning "not is_even"
# Write a function named is_divisible_by_three that takes in a number and returns True or False if that number is a multiple of 3

# Write a function named is_raining that returns True or False whether or not it's currently raining here.

def is_raining():

    return True

    

print(type(is_raining()))



if(is_raining()):

    print("I'll bring an umbrella")

    print("I'll wear sensible shoes")

    print("I will carry my laptop in my backpack")

else:

    print("I'll wear sandals, travel light, and carry my laptop in my hand.")
# Write a function named add_five that takes in a number and returns that number plus five.

# f(x) = x + 5

# Write a function named average that averages a list of number

# Write a function named times_two_plus_ten that returns 2x + 10 

# write a function named square that takes in a number and returns the square of that number

# Write a cube function that returns the number times itself times itself

# write a function named add that takes in two numbers returns the sum of adding both numbers

# Write a function that takes in two numbers, squares each, then sums the result of each square

# sum_of_squares should be your function name

# f(x) = 3x**3 + 2x**2 -7x + 32, name your function y

# write a function called reverse_sign that returns a number multiplied by -1

# f(x) = -x



# write a function named f that takes in a number and multiplies by 10

# write a function named g that takes in a number and divides that number by 37

# write a function named fog that returns f(g(x)) 

# write a function named gof that returns g(f(x))



# Write a function named absolute_value that takes in a number and returns its absolute value
# Write a function named sum_of_cubes that takes in two numbers, cubes each number, then returns the sum of adding the cubed results