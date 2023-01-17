# Example problem:
# Uncomment the line below and run this cell.
# The hashtag "#" character in a line of Python code is the comment character. To "comment" means to add the # and to "uncomment" means to remove the # character.

doing_python_right_now = True

# The lines below will test your answer. If you see an error, then it means that your answer is incorrect or incomplete.
assert doing_python_right_now == True, "If you see a NameError, it means that the variable is not created and assigned a value. An 'Assertion Error' means that the value of the variable is incorrect." 
print("Exercise 0 is correct") # This line will print if your solution passes the assertion above.
# Exercise 1
# On the line below, create a variable named on_mars_right_now and assign it the boolean value of False

on_mars_right_now = False
assert on_mars_right_now == False, "If you see a Name Error, be sure to create the variable and assign it a value."
print("Exercise 1 is correct.")
# Exercise 2
# Create a variable named fruits and assign it a list of fruits containing the following fruit names as strings: 
fruits = ["mango", "banana", "guava", "kiwi", "strawberry"]

assert fruits == ["mango", "banana", "guava", "kiwi", "strawberry"], "If you see an Assert Error, ensure the variable contains all the strings in the provided order"
print("Exercise 2 is correct.")
# Exercise 3
# Create a variable named vegetables and assign it a list of fruits containing the following vegetable names as strings: 
vegetables = ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini"]

assert vegetables == ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini"], "Ensure the variable contains all the strings in the provided order"
print("Exercise 3 is correct.")
# Exercise 4
# Create a variable named numbers and assign it a list of numbers, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

assert numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Ensure the variable contains the numbers 1-10 in order."
print("Exercise 4 is correct.")
# Exercise 5
# Given the following assigment of the list of fruits, add "tomato" to the end of the list. 
fruits = ["mango", "banana", "guava", "kiwi", "strawberry"] + ["tomato"]

assert fruits == ["mango", "banana", "guava", "kiwi", "strawberry", "tomato"], "Ensure the variable contains all the strings in the right order"
print("Exercise 5 is correct")
# Exercise 6
# Given the following assignment of the vegetables list, add "tomato" to the end of the list.
vegetables = ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini"] + ["tomato"]


assert vegetables == ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini", "tomato"], "Ensure the variable contains all the strings in the provided order"
print("Exercise 6 is correct")
# Exercise 7
# Given the list of numbers defined below, reverse the list of numbers that you created above. 
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
numbers.reverse()


assert numbers == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "Assert Error means that the answer is incorrect." 
print("Exercise 7 is correct.")

# Exercise 8
# Sort the vegetables in alphabetical order
vegetables.sort()
assert vegetables == ['broccoli', 'carrot', 'cauliflower', 'eggplant', 'tomato', 'zucchini']
print("Exercise 8 is correct.")
# Exercise 9
# Write the code necessary to sort the fruits in reverse alphabetical order
fruits.sort()
fruits.reverse()
assert fruits == ['tomato', 'strawberry', 'mango', 'kiwi', 'guava', 'banana']
print("Exercise 9 is correct.")
# Exercise 10
# Write the code necessary to produce a single list that holds all fruits then all vegetables in the order as they were sorted above.
fruits_and_veggies = fruits + vegetables

assert fruits_and_veggies == ['tomato', 'strawberry', 'mango', 'kiwi', 'guava', 'banana', 'broccoli', 'carrot', 'cauliflower', 'eggplant', 'tomato', 'zucchini']
print("Exercise 10 is correct")
# Run this cell in order to generate some numbers to use in our functions after this.
import random
    
positive_even_number = random.randrange(2, 101, 2)
negative_even_number = random.randrange(-100, -1, 2)

positive_odd_number = random.randrange(1, 100, 2)
negative_odd_number = random.randrange(-101, 0, 2)
print("We now have some random numbers available for future exercises.")
print("The random positive even number is", positive_even_number)
print("The random positive odd nubmer is", positive_odd_number)
print("The random negative even number", negative_even_number)
print("The random negative odd number", negative_odd_number)

# Example function defintion:
# Write a say_hello function that adds the string "Hello, " to the beginning and "!" to the end of any given input.
def say_hello(name):
    return "Hello, " + name + "!"

assert say_hello("Jane") == "Hello, Jane!", "Double check the inputs and data types"
assert say_hello("Pat") == "Hello, Pat!", "Double check the inputs and data types"
assert say_hello("Astrud") == "Hello, Astrud!", "Double check the inputs and data types"
print("The example function definition ran appropriately")
# Another example function definition:
# This plus_two function takes in a variable and adds 2 to it.
def plus_two(number):
    return number + 2

assert plus_two(3) == 5
assert plus_two(0) == 2
assert plus_two(-2) == 0
print("The plus_two assertions executed appropriately... The second function definition example executed appropriately.")
# Exercise 11
# Write a function definition for a function named add_one that takes in a number and returns that number plus one.
def add_one(x):
    return x + 1
    
assert add_one(2) == 3, "Ensure that the function is defined, named properly, and returns the correct value"
assert add_one(0) == 1, "Zero plus one is one."
assert add_one(positive_even_number) == positive_even_number + 1, "Ensure that the function is defined, named properly, and returns the correct value"
assert add_one(negative_odd_number) == negative_odd_number + 1, "Ensure that the function is defined, named properly, and returns the correct value"
print("Exercise 11 is correct.") 
# Exercise 12
# Write a function definition named is_positive that takes in a number and returns True or False if that number is positive.
def is_positive(n):
    if n > 0:
        return True
    return False
   
assert is_positive(positive_odd_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_positive(positive_even_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_positive(negative_odd_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_positive(negative_even_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
print("Exercise 12 is correct.")
# Exercise 13
# Write a function definition named is_negative that takes in a number and returns True or False if that number is negative.
def is_negative(n):
    if n < 0:
        return True
    return False

assert is_negative(positive_odd_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_negative(positive_even_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_negative(negative_odd_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_negative(negative_even_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
print("Exercise 13 is correct.")
# Exercise 14
# Write a function definition named is_odd that takes in a number and returns True or False if that number is odd.
def is_odd(n):
    if n % 2 != 0:
        return True
    return False
    
assert is_odd(positive_odd_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_odd(positive_even_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_odd(negative_odd_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_odd(negative_even_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
print("Exercise 14 is correct.")
# Exercise 15
# Write a function definition named is_even that takes in a number and returns True or False if that number is even.

def is_even(n):
    if n % 2 == 0:
        return True
    return False

assert is_even(2) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_even(positive_odd_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_even(positive_even_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_even(negative_odd_number) == False, "Ensure that the function is defined, named properly, and returns the correct value"
assert is_even(negative_even_number) == True, "Ensure that the function is defined, named properly, and returns the correct value"
print("Exercise 15 is correct.")
# Exercise 16
# Write a function definition named identity that takes in any argument and returns that argument's value. Don't overthink this one!

def identity(some_value):
    return some_value

assert identity(fruits) == fruits, "Ensure that the function is defined, named properly, and returns the correct value"
assert identity(vegetables) == vegetables, "Ensure that the function is defined, named properly, and returns the correct value"
assert identity(positive_odd_number) == positive_odd_number, "Ensure that the function is defined, named properly, and returns the correct value"
assert identity(positive_even_number) == positive_even_number, "Ensure that the function is defined, named properly, and returns the correct value"
assert identity(negative_odd_number) == negative_odd_number, "Ensure that the function is defined, named properly, and returns the correct value"
assert identity(negative_even_number) == negative_even_number, "Ensure that the function is defined, named properly, and returns the correct value"
print("Exercise 16 is correct.")
# Exercise 17
# Write a function definition named is_positive_odd that takes in a number and returns True or False if the value is both greater than zero and odd
def is_positive_odd(n):
    if n > 0 and n % 2 != 0:
        return True
    return False

assert is_positive_odd(3) == True, "Double check your syntax and logic" 
assert is_positive_odd(positive_odd_number) == True, "Double check your syntax and logic"
assert is_positive_odd(positive_even_number) == False, "Double check your syntax and logic"
assert is_positive_odd(negative_odd_number) == False, "Double check your syntax and logic"
assert is_positive_odd(negative_even_number) == False, "Double check your syntax and logic"
print("Exercise 17 is correct.")
# Exercise 18
# Write a function definition named is_positive_even that takes in a number and returns True or False if the value is both greater than zero and even
def is_positive_even(n):
    if n > 0 and n % 2 == 0:
        return True
    return False


assert is_positive_even(4) == True, "Double check your syntax and logic" 
assert is_positive_even(positive_odd_number) == False, "Double check your syntax and logic"
assert is_positive_even(positive_even_number) == True, "Double check your syntax and logic"
assert is_positive_even(negative_odd_number) == False, "Double check your syntax and logic"
assert is_positive_even(negative_even_number) == False, "Double check your syntax and logic"
print("Exercise 18 is correct.")
# Exercise 19
# Write a function definition named is_negative_odd that takes in a number and returns True or False if the value is both less than zero and odd.
def is_negative_odd(n):
    if n < 0 and n % 2 != 0:
        return True
    return False


assert is_negative_odd(-3) == True, "Double check your syntax and logic" 
assert is_negative_odd(positive_odd_number) == False, "Double check your syntax and logic"
assert is_negative_odd(positive_even_number) == False, "Double check your syntax and logic"
assert is_negative_odd(negative_odd_number) == True, "Double check your syntax and logic"
assert is_negative_odd(negative_even_number) == False, "Double check your syntax and logic"
print("Exercise 19 is correct.")

# Exercise 20
# Write a function definition named is_negative_even that takes in a number and returns True or False if the value is both less than zero and even.
def is_negative_even(n):
    if n < 0 and n % 2 == 0:
        return True
    return False

assert is_negative_even(-4) == True, "Double check your syntax and logic" 
assert is_negative_even(positive_odd_number) == False, "Double check your syntax and logic"
assert is_negative_even(positive_even_number) == False, "Double check your syntax and logic"
assert is_negative_even(negative_odd_number) == False, "Double check your syntax and logic"
assert is_negative_even(negative_even_number) == True, "Double check your syntax and logic"
print("Exercise 20 is correct.")
# Exercise 21
# Write a function definition named half that takes in a number and returns half the provided number.
def half(n):
    return n/2

assert half(4) == 2
assert half(5) == 2.5
assert half(positive_odd_number) == positive_odd_number / 2
assert half(positive_even_number) == positive_even_number / 2
assert half(negative_odd_number) == negative_odd_number / 2
assert half(negative_even_number) == negative_even_number / 2
print("Exercise 21 is correct.")
# Exercise 22
# Write a function definition named double that takes in a number and returns double the provided number.
def double(n):
    return n * 2


assert double(4) == 8
assert double(5) == 10
assert double(positive_odd_number) == positive_odd_number * 2
assert double(positive_even_number) == positive_even_number * 2
assert double(negative_odd_number) == negative_odd_number * 2
assert double(negative_even_number) == negative_even_number * 2
print("Exercise 22 is correct.")
# Exercise 23
# Write a function definition named triple that takes in a number and returns triple the provided number.
def triple(n):
    return n * 3

assert triple(4) == 12
assert triple(5) == 15
assert triple(positive_odd_number) == positive_odd_number * 3
assert triple(positive_even_number) == positive_even_number * 3
assert triple(negative_odd_number) == negative_odd_number * 3
assert triple(negative_even_number) == negative_even_number * 3
print("Exercise 23 is correct.")
# Exercise 24
# Write a function definition named reverse_sign that takes in a number and returns the provided number but with the sign reversed.
def reverse_sign(n):
    return n * -1

assert reverse_sign(4) == -4
assert reverse_sign(-5) == 5
assert reverse_sign(positive_odd_number) == positive_odd_number * -1
assert reverse_sign(positive_even_number) == positive_even_number * -1
assert reverse_sign(negative_odd_number) == negative_odd_number * -1
assert reverse_sign(negative_even_number) == negative_even_number * -1
print("Exercise 24 is correct.")
# Exercise 25
# Write a function definition named absolute_value that takes in a number and returns the absolute value of the provided number
def absolute_value(n):
    return abs(n)

assert absolute_value(4) == 4
assert absolute_value(-5) == 5
assert absolute_value(positive_odd_number) == positive_odd_number
assert absolute_value(positive_even_number) == positive_even_number
assert absolute_value(negative_odd_number) == negative_odd_number * -1
assert absolute_value(negative_even_number) == negative_even_number * -1
print("Exercise 25 is correct.")
# Exercise 26
# Write a function definition named is_multiple_of_three that takes in a number and returns True or False if the number is evenly divisible by 3.
def is_multiple_of_three(n):
    if n % 3 == 0:
        return True
    return False

assert is_multiple_of_three(3) == True
assert is_multiple_of_three(15) == True
assert is_multiple_of_three(9) == True
assert is_multiple_of_three(4) == False
assert is_multiple_of_three(10) == False
print("Exercise 26 is correct.")
# Exercise 27
# Write a function definition named is_multiple_of_fivethat takes in a number and returns True or False if the number is evenly divisible by 5.
def is_multiple_of_five(n):
    if n % 5 == 0:
        return True
    return False

assert is_multiple_of_five(3) == False
assert is_multiple_of_five(15) == True
assert is_multiple_of_five(9) == False
assert is_multiple_of_five(4) == False
assert is_multiple_of_five(10) == True
print("Exercise 27 is correct.")
# Exercise 28
# Write a function definition named is_multiple_of_both_three_and_five that takes in a number and returns True or False if the number is evenly divisible by both 3 and 5.
def is_multiple_of_both_three_and_five(n):
    if n % 3 == 0 and n % 5 == 0:
        return True
    return False


assert is_multiple_of_both_three_and_five(15) == True
assert is_multiple_of_both_three_and_five(45) == True
assert is_multiple_of_both_three_and_five(3) == False
assert is_multiple_of_both_three_and_five(9) == False
assert is_multiple_of_both_three_and_five(4) == False
print("Exercise 28 is correct.")
# Exercise 29
# Write a function definition named square that takes in a number and returns the number times itself.
def square(n):
    return n**2
assert square(3) == 9
assert square(2) == 4
assert square(9) == 81
assert square(positive_odd_number) == positive_odd_number * positive_odd_number
print("Exercise 29 is correct.")
# Exercise 30
# Write a function definition named add that takes in two numbers and returns the sum.
def add(x, y):
    return x + y

assert add(3, 2) == 5
assert add(10, -2) == 8
assert add(5, 7) == 12
print("Exercise 30 is correct.")
# Exercise 31
# Write a function definition named cube that takes in a number and returns the number times itself, times itself.
def cube(n):
    return n**3

assert cube(3) == 27
assert cube(2) == 8
assert cube(5) == 125
assert cube(positive_odd_number) == positive_odd_number * positive_odd_number * positive_odd_number
print("Exercise 31 is correct.")
# Exercise 32
# Write a function definition named square_root that takes in a number and returns the square root of the provided number
import math as math
def square_root(n):
    return math.sqrt(n)

assert square_root(4) == 2.0
assert square_root(64) == 8.0
assert square_root(81) == 9.0
print("Exercise 32 is correct.")
# Exercise 33
# Write a function definition named subtract that takes in two numbers and returns the first minus the second argument.
def subtract(x, y):
    return x - y

assert subtract(8, 6) == 2
assert subtract(27, 4) == 23
assert subtract(12, 2) == 10
print("Exercise 33 is correct.")
# Exercise 34
# Write a function definition named multiply that takes in two numbers and returns the first times the second argument.
def multiply(x, y):
    return x * y

assert multiply(2, 1) == 2
assert multiply(3, 5) == 15
assert multiply(5, 2) == 10
print("Exercise 34 is correct.")
# Exercise 35
# Write a function definition named divide that takes in two numbers and returns the first argument divided by the second argument.
def divide(x,y):
    return x/y

assert divide(27, 9) == 3
assert divide(15, 3) == 5
assert divide(5, 2) == 2.5
assert divide(10, 2) == 5
print("Exercise 35 is correct.")
# Exercise 36
# Write a function definition named quotient that takes in two numbers and returns only the quotient from dividing the first argument by the second argument.
def quotient(x, y):
    return x//y

assert quotient(27, 9) == 3
assert quotient(5, 2) == 2
assert quotient(10, 3) == 3
print("Exercise 36 is correct.")
# Exercise 37
# Write a function definition named remainder that takes in two numbers and returns the remainder of first argument divided by the second argument.
def remainder(x, y):
    return x % y 

assert remainder(3, 3) == 0
assert remainder(5, 2) == 1
assert remainder(7, 5) == 2
print("Exercise 37 is correct.")
# Exercise 38
# Write a function definition named sum_of_squares that takes in two numbers, squares each number, then returns the sum of both squares.
def sum_of_squares(x, y):
    return x**2 + y**2

assert sum_of_squares(3, 2) == 13
assert sum_of_squares(5, 2) == 29
assert sum_of_squares(2, 4) == 20
print("Exercise 38 is correct.")
# Exercise 39
# Write a function definition named times_two_plus_three that takes in a number, multiplies it by two, adds 3 and returns the result.
def times_two_plus_three(n):
    return n * 2 + 3

assert times_two_plus_three(0) == 3
assert times_two_plus_three(1) == 5
assert times_two_plus_three(2) == 7
assert times_two_plus_three(3) == 9
assert times_two_plus_three(5) == 13
print("Exercise 39 is correct.")
# Exercise 40
# Write a function definition named area_of_rectangle that takes in two numbers and returns the product.
def area_of_rectangle(x, y):
    return x * y

assert area_of_rectangle(1, 3) == 3
assert area_of_rectangle(5, 2) == 10
assert area_of_rectangle(2, 7) == 14
assert area_of_rectangle(5.3, 10.3) == 54.59
print("Exercise 40 is correct.")
import math
# Exercise 41
# Write a function definition named area_of_circle that takes in a number representing a circle's radius and returns the area of the circl
def area_of_circle(n):
    return math.pi * n**2

assert area_of_circle(3) == 28.274333882308138
assert area_of_circle(5) == 78.53981633974483
assert area_of_circle(7) == 153.93804002589985
print("Exercise 41 is correct.")
import math
# Exercise 42
# Write a function definition named circumference that takes in a number representing a circle's radius and returns the circumference.
def circumference(n):
    return 2 * math.pi * n

assert circumference(3) == 18.84955592153876
assert circumference(5) == 31.41592653589793
assert circumference(7) == 43.982297150257104
print("Exercise 42 is correct.")
# Exercise 43
# Write a function definition named is_vowel that takes in value and returns True if the value is a, e, i, o, u in upper or lower case.
def is_vowel(w):
    vowels = ["a", "e", "i", "o", "u"]  
    for vowel in vowels:
        if w.lower() == vowel:
            return True
    return False
assert is_vowel("a") == True
assert is_vowel("U") == True
assert is_vowel("banana") == False
assert is_vowel("Q") == False
assert is_vowel("y") == False
print("Exercise 43 is correct.")
# Exercise 44
# Write a function definition named has_vowels that takes in value and returns True if the string contains any vowels.
def has_vowels(something):
    letters = list(something)
    for letter in letters:
        if is_vowel(letter):
            return True
    return False

assert has_vowels("banana") == True
assert has_vowels("ubuntu") == True
assert has_vowels("QQQQ") == False
assert has_vowels("wyrd") == False
print("Exercise 44 is correct.")
# Exercise 45
# Write a function definition named count_vowels that takes in value and returns the count of the nubmer of vowels in a sequence.
def count_vowels(something):
    countvs = []
    letters = list(something)
    for letter in letters:
        if is_vowel(letter):
            countvs.append(letter)
    return len(countvs)

assert count_vowels("banana") == 3
assert count_vowels("ubuntu") == 3
assert count_vowels("mango") == 2
assert count_vowels("QQQQ") == 0
assert count_vowels("wyrd") == 0
print("Exercise 45 is correct.")
# Exercise 46
# Write a function definition named remove_vowels that takes in string and returns the string without any vowels
def remove_vowels(something):
    letters = list(something)
    for letter in letters:
        if is_vowel(letter):
            letters.remove(letter)
    return ''.join(letters)

assert remove_vowels("banana") == "bnn"
assert remove_vowels("ubuntu") == "bnt"
assert remove_vowels("mango") == "mng"
assert remove_vowels("QQQQ") == "QQQQ"
print("Exercise 46 is correct.")
# Exercise 47
# Write a function definition named starts_with_vowel that takes in string and True if the string starts with a vowel
def starts_with_vowel(something):
    letters = list(something)
    if is_vowel(letters[0]):
        return True
    return False

assert starts_with_vowel("ubuntu") == True
assert starts_with_vowel("banana") == False
assert starts_with_vowel("mango") == False
print("Exercise 47 is correct.")
# Exercise 48
# Write a function definition named ends_with_vowel that takes in string and True if the string ends with a vowel
def ends_with_vowel(something):
    letters = list(something)
    if is_vowel(letters[-1]):
        return True
    return False

assert ends_with_vowel("ubuntu") == True
assert ends_with_vowel("banana") == True
assert ends_with_vowel("mango") == True
assert ends_with_vowel("spinach") == False
print("Exercise 48 is correct.")
# Exercise 49
# Write a function definition named starts_and_ends_with_vowel that takes in string and returns True if the string starts and ends with a vowel
def starts_and_ends_with_vowel(something):
    letters = list(something)
    if is_vowel(letters[-1]) and is_vowel(letters[0]):
        return True
    return False

assert starts_and_ends_with_vowel("ubuntu") == True
assert starts_and_ends_with_vowel("banana") == False
assert starts_and_ends_with_vowel("mango") == False
print("Exercise 49 is correct.")
# Exercise 50
# Write a function definition named first that takes in sequence and returns the first value of that sequence.
def first(something):
    return something[0]

assert first("ubuntu") == "u"
assert first([1, 2, 3]) == 1
assert first(["python", "is", "awesome"]) == "python"
print("Exercise 50 is correct.")
# Exercise 51
# Write a function definition named second that takes in sequence and returns the second value of that sequence.
def second(something):
    return something[1]

assert second("ubuntu") == "b"
assert second([1, 2, 3]) == 2
assert second(["python", "is", "awesome"]) == "is"
print("Exercise 51 is correct.")
# Exercise 52
# Write a function definition named third that takes in sequence and returns the third value of that sequence.
def third(something):
    return something[2]

assert third("ubuntu") == "u"
assert third([1, 2, 3]) == 3
assert third(["python", "is", "awesome"]) == "awesome"
print("Exercise 52 is correct.")
# Exercise 53
# Write a function definition named forth that takes in sequence and returns the forth value of that sequence.
def forth(something):
    return something[3]

assert forth("ubuntu") == "n"
assert forth([1, 2, 3, 4]) == 4
assert forth(["python", "is", "awesome", "right?"]) == "right?"
print("Exercise 53 is correct.")
# Exercise 54
# Write a function definition named last that takes in sequence and returns the last value of that sequence.
def last(something):
    return something[-1]

assert last("ubuntu") == "u"
assert last([1, 2, 3, 4]) == 4
assert last(["python", "is", "awesome"]) == "awesome"
assert last(["kiwi", "mango", "guava"]) == "guava"
print("Exercise 54 is correct.")
# Exercise 55
# Write a function definition named second_to_last that takes in sequence and returns the second to last value of that sequence.
def second_to_last(something):
    return something[-2]

assert second_to_last("ubuntu") == "t"
assert second_to_last([1, 2, 3, 4]) == 3
assert second_to_last(["python", "is", "awesome"]) == "is"
assert second_to_last(["kiwi", "mango", "guava"]) == "mango"
print("Exercise 55 is correct.")
# Exercise 56
# Write a function definition named third_to_last that takes in sequence and returns the third to last value of that sequence.
def third_to_last(something):
    return something[-3]

assert third_to_last("ubuntu") == "n"
assert third_to_last([1, 2, 3, 4]) == 2
assert third_to_last(["python", "is", "awesome"]) == "python"
assert third_to_last(["strawberry", "kiwi", "mango", "guava"]) == "kiwi"
print("Exercise 56 is correct.")
# Exercise 57
# Write a function definition named first_and_second that takes in sequence and returns the first and second value of that sequence as a list
def first_and_second(something):
    return something[:2]

assert first_and_second([1, 2, 3, 4]) == [1, 2]
assert first_and_second(["python", "is", "awesome"]) == ["python", "is"]
assert first_and_second(["strawberry", "kiwi", "mango", "guava"]) == ["strawberry", "kiwi"]
print("Exercise 57 is correct.")
list1 = [1, 2, 3]
str1 = ''.join(str(e) for e in list1)
newlist = [list1[0], list1[-1]]
print(newlist)
# Exercise 58
# Write a function definition named first_and_last that takes in sequence and returns the first and last value of that sequence as a list
def first_and_last(something):
    contain = list(something)
    newlist = [contain[0], contain[-1]]
    return newlist  
assert first_and_last([1, 2, 3, 4]) == [1, 4]
assert first_and_last(["python", "is", "awesome"]) == ["python", "awesome"]
assert first_and_last(["strawberry", "kiwi", "mango", "guava"]) == ["strawberry", "guava"]
print("Exercise 58 is correct.")
# Exercise 59
# Write a function definition named first_to_last that takes in sequence and returns the sequence with the first value moved to the end of the sequence.
def first_to_last(something):
    contain = list(something)
    first = contain[0]
    contain.append(first)
    return contain[1:]

assert first_to_last([1, 2, 3, 4]) == [2, 3, 4, 1]
assert first_to_last(["python", "is", "awesome"]) == ["is", "awesome", "python"]
assert first_to_last(["strawberry", "kiwi", "mango", "guava"]) == ["kiwi", "mango", "guava", "strawberry"]
print("Exercise 59 is correct.")
# Exercise 60
# Write a function definition named sum_all that takes in sequence of numbers and returns all the numbers added together.
def sum_all(something):
    return sum(something)

assert sum_all([1, 2, 3, 4]) == 10
assert sum_all([3, 3, 3]) == 9
assert sum_all([0, 5, 6]) == 11
print("Exercise 60 is correct.")
# Exercise 61
# Write a function definition named mean that takes in sequence of numbers and returns the average value
def mean(something):
    return sum(something)/len(list(something))

assert mean([1, 2, 3, 4]) == 2.5
assert mean([3, 3, 3]) == 3
assert mean([1, 5, 6]) == 4
print("Exercise 61 is correct.")
# Exercise 62
# Write a function definition named median that takes in sequence of numbers and returns the average value
def median(something):
    n_num = list(something)
    n = len(n_num) 
    n_num.sort() 
  
    if n % 2 == 0: 
        median1 = n_num[n//2] 
        median2 = n_num[n//2 - 1] 
        median = (median1 + median2)/2
    else: 
        median = n_num[n//2] 
    return median

assert median([1, 2, 3, 4, 5]) == 3.0
assert median([1, 2, 3]) == 2.0
assert median([1, 5, 6]) == 5.0
assert median([1, 2, 5, 6]) == 3.5
print("Exercise 62 is correct.")
# Exercise 63
# Write a function definition named mode that takes in sequence of numbers and returns the most commonly occuring value
def mode(something):
    return max(set(something), key=something.count)

assert mode([1, 2, 2, 3, 4]) == 2
assert mode([1, 1, 2, 3]) == 1
assert mode([2, 2, 3, 3, 3]) == 3
print("Exercise 63 is correct.")
# Exercise 64
# Write a function definition named product_of_all that takes in sequence of numbers and returns the product of multiplying all the numbers together
def product_of_all(somethings):
    product = 1
    for something in somethings: 
         product = product * something
    return product 

assert product_of_all([1, 2, 3]) == 6
assert product_of_all([3, 4, 5]) == 60
assert product_of_all([2, 2, 3, 0]) == 0
print("Exercise 64 is correct.")
# Run this cell in order to use the following list of numbers for the next exercises
numbers = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5] 
# Exercise 65
# Write a function definition named get_highest_number that takes in sequence of numbers and returns the largest number.
def get_highest_number(something):
    numbers = list(something)
    numbers.sort(reverse = True)
    return numbers[0]

assert get_highest_number([1, 2, 3]) == 3
assert get_highest_number([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) == 5
assert get_highest_number([-5, -3, 1]) == 1
print("Exercise 65 is correct.")
# Exercise 66
# Write a function definition named get_smallest_number that takes in sequence of numbers and returns the smallest number.
def get_smallest_number(something):
    numbers = list(something)
    numbers.sort()
    return numbers[0]

assert get_smallest_number([1, 2, 3]) == 1
assert get_smallest_number([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) == -5
assert get_smallest_number([-4, -3, 1]) == -4
print("Exercise 66 is correct.")
# Exercise 67
# Write a function definition named only_odd_numbers that takes in sequence of numbers and returns the odd numbers in a list.
def only_odd_numbers(something):
    numbers = list(something)
    odds = []
    for number in numbers:
        if number % 2 != 0:
            odds.append(number)
    return odds

assert only_odd_numbers([1, 2, 3]) == [1, 3]
assert only_odd_numbers([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) == [-5, -3, -1, 1, 3, 5]
assert only_odd_numbers([-4, -3, 1]) == [-3, 1]
print("Exercise 67 is correct.")
# Exercise 68
# Write a function definition named only_even_numbers that takes in sequence of numbers and returns the even numbers in a list.
def only_even_numbers(something):
    numbers = list(something)
    evens = []
    for number in numbers:
        if number % 2 == 0:
            evens.append(number)
    return evens

assert only_even_numbers([1, 2, 3]) == [2]
assert only_even_numbers([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) == [-4, -2, 2, 4]
assert only_even_numbers([-4, -3, 1]) == [-4]
print("Exercise 68 is correct.")
# Exercise 69
# Write a function definition named only_positive_numbers that takes in sequence of numbers and returns the positive numbers in a list.
def only_positive_numbers(something):
    numbers = list(something)
    posnums = []
    for number in numbers:
        if number > 0:
            posnums.append(number)
    return posnums

assert only_positive_numbers([1, 2, 3]) == [1, 2, 3]
assert only_positive_numbers([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
assert only_positive_numbers([-4, -3, 1]) == [1]
print("Exercise 69 is correct.")
# Exercise 70
# Write a function definition named only_negative_numbers that takes in sequence of numbers and returns the negative numbers in a list.
def only_negative_numbers(something):
    numbers = list(something)
    negnums = []
    for number in numbers:
        if number < 0:
            negnums.append(number)
    return negnums


assert only_negative_numbers([1, 2, 3]) == []
assert only_negative_numbers([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) == [-5, -4, -3, -2, -1]
assert only_negative_numbers([-4, -3, 1]) == [-4, -3]
print("Exercise 70 is correct.")
# Exercise 71
# Write a function definition named has_evens that takes in sequence of numbers and returns True if there are any even numbers in the sequence
def has_evens(something):
    numbers = list(something)
    for number in numbers:
        if number % 2 == 0:
            return True
    return False

assert has_evens([1, 2, 3]) == True
assert has_evens([2, 5, 6]) == True
assert has_evens([3, 3, 3]) == False
assert has_evens([]) == False
print("Exercise 71 is correct.")
# Exercise 72
# Write a function definition named count_evens that takes in sequence of numbers and returns the number of even numbers
def count_evens(something):
    count = only_even_numbers(something)
    return len(count)

assert count_evens([1, 2, 3]) == 1
assert count_evens([2, 5, 6]) == 2
assert count_evens([3, 3, 3]) == 0
assert count_evens([5, 6, 7, 8] ) == 2
print("Exercise 72 is correct.")
# Exercise 73
# Write a function definition named has_odds that takes in sequence of numbers and returns True if there are any odd numbers in the sequence
def has_odds(something):
    numbers = list(something)
    for number in numbers:
        if number % 2 != 0:
            return True
    return False

assert has_odds([1, 2, 3]) == True
assert has_odds([2, 5, 6]) == True
assert has_odds([3, 3, 3]) == True
assert has_odds([2, 4, 6]) == False
print("Exercise 73 is correct.")
# Exercise 74
# Write a function definition named count_odds that takes in sequence of numbers and returns True if there are any odd numbers in the sequence
def count_odds(something):
    count = only_odd_numbers(something)
    return len(count)
    

assert count_odds([1, 2, 3]) == 2
assert count_odds([2, 5, 6]) == 1
assert count_odds([3, 3, 3]) == 3
assert count_odds([2, 4, 6]) == 0
print("Exercise 74 is correct.")
# Exercise 75
# Write a function definition named count_negatives that takes in sequence of numbers and returns a count of the number of negative numbers
def count_negatives(something):
    count = only_negative_numbers(something)
    return len(count)

assert count_negatives([1, -2, 3]) == 1
assert count_negatives([2, -5, -6]) == 2
assert count_negatives([3, 3, 3]) == 0
print("Exercise 75 is correct.")
# Exercise 76
# Write a function definition named count_positives that takes in sequence of numbers and returns a count of the number of positive numbers
def count_positives(something):
    count = only_positive_numbers(something)
    return len(count)

assert count_positives([1, -2, 3]) == 2
assert count_positives([2, -5, -6]) == 1
assert count_positives([3, 3, 3]) == 3
assert count_positives([-2, -1, -5]) == 0
print("Exercise 76 is correct.")
# Exercise 77
# Write a function definition named only_positive_evens that takes in sequence of numbers and returns a list containing all the positive evens from the sequence
def only_positive_evens(something):
    numbers = list(something)
    poseven = []
    for number in numbers:
        if number % 2 == 0 and number > 0:
            poseven.append(number)
    return poseven

assert only_positive_evens([1, -2, 3]) == []
assert only_positive_evens([2, -5, -6]) == [2]
assert only_positive_evens([3, 3, 4, 6]) == [4, 6]
assert only_positive_evens([2, 3, 4, -1, -5]) == [2, 4]
print("Exercise 77 is correct.")
# Exercise 78
# Write a function definition named only_positive_odds that takes in sequence of numbers and returns a list containing all the positive odd numbers from the sequence
def only_positive_odds(something):
    numbers = list(something)
    posodds = []
    for number in numbers:
        if number % 2 != 0 and number > 0:
            posodds.append(number)
    return posodds

assert only_positive_odds([1, -2, 3]) == [1, 3]
assert only_positive_odds([2, -5, -6]) == []
assert only_positive_odds([3, 3, 4, 6]) == [3, 3]
assert only_positive_odds([2, 3, 4, -1, -5]) == [3]
print("Exercise 78 is correct.")
# Exercise 79
# Write a function definition named only_negative_evens that takes in sequence of numbers and returns a list containing all the negative even numbers from the sequence
def only_negative_evens(something):
    numbers = list(something)
    negevens = []
    for number in numbers:
        if number % 2 == 0 and number < 0:
            negevens.append(number)
    return negevens

assert only_negative_evens([1, -2, 3]) == [-2]
assert only_negative_evens([2, -5, -6]) == [-6]
assert only_negative_evens([3, 3, 4, 6]) == []
assert only_negative_evens([-2, 3, 4, -1, -4]) == [-2, -4]
print("Exercise 79 is correct.")
# Exercise 80
# Write a function definition named only_negative_odds that takes in sequence of numbers and returns a list containing all the negative odd numbers from the sequence
def only_negative_odds(something):
    numbers = list(something)
    negodds = []
    for number in numbers:
        if number % 2 != 0 and number < 0:
            negodds.append(number)
    return negodds

assert only_negative_odds([1, -2, 3]) == []
assert only_negative_odds([2, -5, -6]) == [-5]
assert only_negative_odds([3, 3, 4, 6]) == []
assert only_negative_odds([2, -3, 4, -1, -4]) == [-3, -1]
print("Exercise 80 is correct.")
# Exercise 81
# Write a function definition named shortest_string that takes in a list of strings and returns the shortest string in the list.
def shortest_string(something):
    texts = list(something)
    nums = []
    for text in texts:
        nums.append(len(text))
    nums = min(nums)
    for text in texts:
        if len(text) == nums:
            return text

assert shortest_string(["kiwi", "mango", "strawberry"]) == "kiwi"
assert shortest_string(["hello", "everybody"]) == "hello"
assert shortest_string(["mary", "had", "a", "little", "lamb"]) == "a"
print("Exercise 81 is correct.")
# Exercise 82
# Write a function definition named longest_string that takes in sequence of strings and returns the longest string in the list.
def longest_string(something):
    texts = list(something)
    nums = []
    for text in texts:
        nums.append(len(text))
    nums = max(nums)
    for text in texts:
        if len(text) == nums:
            return text

assert longest_string(["kiwi", "mango", "strawberry"]) == "strawberry"
assert longest_string(["hello", "everybody"]) == "everybody"
assert longest_string(["mary", "had", "a", "little", "lamb"]) == "little"
print("Exercise 82 is correct.")
# Example set function usage
print(set("kiwi"))
print(set([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
# Exercise 83
# Write a function definition named get_unique_values that takes in a list and returns a set with only the unique values from that list.
def get_unique_values(something):
    return set(something)

assert get_unique_values(["ant", "ant", "mosquito", "mosquito", "ladybug"]) == {"ant", "mosquito", "ladybug"}
assert get_unique_values(["b", "a", "n", "a", "n", "a", "s"]) == {"b", "a", "n", "s"}
assert get_unique_values(["mary", "had", "a", "little", "lamb", "little", "lamb", "little", "lamb"]) == {"mary", "had", "a", "little", "lamb"}
print("Exercise 83 is correct.")
# Exercise 84
# Write a function definition named get_unique_values_from_two_lists that takes two lists and returns a single set with only the unique values
def get_unique_values_from_two_lists(list1, list2):
    full = list1 + list2
    return set(full)

assert get_unique_values_from_two_lists([5, 1, 2, 3], [3, 4, 5, 5]) == {1, 2, 3, 4, 5}
assert get_unique_values_from_two_lists([1, 1], [2, 2, 3]) == {1, 2, 3}
assert get_unique_values_from_two_lists(["tomato", "mango", "kiwi"], ["eggplant", "tomato", "broccoli"]) == {"tomato", "mango", "kiwi", "eggplant", "broccoli"}
print("Exercise 84 is correct.")
# Exercise 85
# Write a function definition named get_values_in_common that takes two lists and returns a single set with the values that each list has in common
def get_values_in_common(list1, list2):
    fsts = set(list1)
    snds = set(list2)
    fulls = []
    endlist = []
    for fst in fsts:
        fulls.append(fst)
    for snd in snds:
        fulls.append(snd)
    for full in fulls:
        if fulls.count(full) > 1:
            endlist.append(full)
    return set(endlist)
    
assert get_values_in_common([5, 1, 2, 3], [3, 4, 5, 5]) == {3, 5}
assert get_values_in_common([1, 2], [2, 2, 3]) == {2}
assert get_values_in_common(["tomato", "mango", "kiwi"], ["eggplant", "tomato", "broccoli"]) == {"tomato"}
print("Exercise 85 is correct.")
# Exercise 86
# Write a function definition named get_values_not_in_common that takes two lists and returns a single set with the values that each list does not have in common
def get_values_not_in_common(list1, list2):
    fsts = set(list1)
    snds = set(list2)
    fulls = []
    endlist = []
    for fst in fsts:
        fulls.append(fst)
    for snd in snds:
        fulls.append(snd)
    for full in fulls:
        if fulls.count(full) == 1:
            endlist.append(full)
    return set(endlist)

assert get_values_not_in_common([5, 1, 2, 3], [3, 4, 5, 5]) == {1, 2, 4}
assert get_values_not_in_common([1, 1], [2, 2, 3]) == {1, 2, 3}
assert get_values_not_in_common(["tomato", "mango", "kiwi"], ["eggplant", "tomato", "broccoli"]) == {"mango", "kiwi", "eggplant", "broccoli"}
print("Exercise 86 is correct.")
# Run this cell in order to have these two dictionary variables defined.
tukey_paper = {
    "title": "The Future of Data Analysis",
    "author": "John W. Tukey",
    "link": "https://projecteuclid.org/euclid.aoms/1177704711",
    "year_published": 1962
}

thomas_paper = {
    "title": "A mathematical model of glutathione metabolism",
    "author": "Rachel Thomas",
    "link": "https://www.ncbi.nlm.nih.gov/pubmed/18442411",
    "year_published": 2008
}
# Exercise 87
# Write a function named get_paper_title that takes in a dictionary and returns the title property
def get_paper_title(something):
    return something['title']


assert get_paper_title(tukey_paper) == "The Future of Data Analysis"
assert get_paper_title(thomas_paper) == "A mathematical model of glutathione metabolism"
print("Exercise 87 is correct.")
# Exercise 88
# Write a function named get_year_published that takes in a dictionary and returns the value behind the "year_published" key.
def get_year_published(something):
    return something['year_published']

assert get_year_published(tukey_paper) == 1962
assert get_year_published(thomas_paper) == 2008
print("Exercise 88 is correct.")
# Run this code to create data for the next two questions
book = {
    "title": "Genetic Algorithms and Machine Learning for Programmers",
    "price": 36.99,
    "author": "Frances Buontempo"
}
# Exercise 89
# Write a function named get_price that takes in a dictionary and returns the price
def get_price(something):
    return something['price']

assert get_price(book) == 36.99
print("Exercise 89 is complete.")
# Exercise 90
# Write a function named get_book_author that takes in a dictionary (the above declared book variable) and returns the author's name
def get_book_author(something):
    return something['author']

assert get_book_author(book) == "Frances Buontempo"
print("Exercise 90 is complete.")
# Run this cell in order to have some setup data for the next exercises
books = [
    {
        "title": "Genetic Algorithms and Machine Learning for Programmers",
        "price": 36.99,
        "author": "Frances Buontempo"
    },
    {
        "title": "The Visual Display of Quantitative Information",
        "price": 38.00,
        "author": "Edward Tufte"
    },
    {
        "title": "Practical Object-Oriented Design",
        "author": "Sandi Metz",
        "price": 30.47
    },
    {
        "title": "Weapons of Math Destruction",
        "author": "Cathy O'Neil",
        "price": 17.44
    }
]
# Exercise 91
# Write a function named get_number_of_books that takes in a list of objects and returns the number of dictionaries in that list.
def get_number_of_books(something):
    return len(something)

assert get_number_of_books(books) == 4
print("Exercise 91 is complete.")
# Exercise 92
# Write a function named total_of_book_prices that takes in a list of dictionaries and returns the sum total of all the book prices added together
def total_of_book_prices(something):
    prices = []
    for d in something:
         prices.append(d['price'])
    return sum(prices)

assert total_of_book_prices(books) == 122.9
print("Exercise 92 is complete.")
# Exercise 93
# Write a function named get_average_book_price that takes in a list of dictionaries and returns the average book price.
def get_average_book_price(something):
    prices = []
    for d in something:
        prices.append(d['price'])
    return sum(prices)/len(prices)

assert get_average_book_price(books) == 30.725
print("Exercise 93 is complete.")
# Exercise 94
# Write a function called highest_priced_book that takes in the above defined list of dictionaries "books" and returns the dictionary containing the title, price, and author of the book with the highest priced book.
# Hint: Much like sometimes start functions with a variable set to zero, you may want to create a dictionary with the price set to zero to compare to each dictionary's price in the list
def highest_price_book(something):
    prices = []
    for d in something:
        prices.append(d['price'])
    maxprice = max(prices)
    for d in something:
        if d['price'] == maxprice:
            return d
    
assert highest_price_book(books) == {
    "title": "The Visual Display of Quantitative Information",
    "price": 38.00,
    "author": "Edward Tufte"
}

print("Exercise 94 is complete")
# Exercise 95
# Write a function called lowest_priced_book that takes in the above defined list of dictionaries "books" and returns the dictionary containing the title, price, and author of the book with the lowest priced book.
# Hint: Much like sometimes start functions with a variable set to zero or float('inf'), you may want to create a dictionary with the price set to float('inf') to compare to each dictionary in the list
def lowest_price_book(something):
    prices = []
    for d in something:
        prices.append(d['price'])
    lowest = min(prices)
    for d in something:
        if d['price'] == lowest:
            return d

assert lowest_price_book(books) == {
    "title": "Weapons of Math Destruction",
    "author": "Cathy O'Neil",
    "price": 17.44
}
print("Exercise 95 is complete.")
shopping_cart = {
    "tax": .08,
    "items": [
        {
            "title": "orange juice",
            "price": 3.99,
            "quantity": 1
        },
        {
            "title": "rice",
            "price": 1.99,
            "quantity": 3
        },
        {
            "title": "beans",
            "price": 0.99,
            "quantity": 3
        },
        {
            "title": "chili sauce",
            "price": 2.99,
            "quantity": 1
        },
        {
            "title": "chocolate",
            "price": 0.75,
            "quantity": 9
        }
    ]
}
# Exercise 96
# Write a function named get_tax_rate that takes in the above shopping cart as input and returns the tax rate.
# Hint: How do you access a key's value on a dictionary? The tax rate is one key of the entire shopping_cart dictionary.
def get_tax_rate(something):
    return something['tax']

assert get_tax_rate(shopping_cart) == .08
print("Exercise 96 is complete")
# Exercise 97
# Write a function named number_of_item_types that takes in the shopping cart as input and returns the number of unique item types in the shopping cart. 
# We're not yet using the quantity of each item, but rather focusing on determining how many different types of items are in the cart.
def number_of_item_types(something):
    return len(something['items'])

assert number_of_item_types(shopping_cart) == 5
print("Exercise 97 is complete.")
# Exercise 98
# Write a function named total_number_of_items that takes in the shopping cart as input and returns the total number all item quantities.
# This should return the sum of all of the quantities from each item type
def total_number_of_items(cart):
    items = cart['items']
    total = 0
    for item in items:
        total = total + item['quantity']
    return total
            
        
assert total_number_of_items(shopping_cart) == 17
print("Exercise 98 is complete.")
# Exercise 99
# Write a function named get_average_item_price that takes in the shopping cart as an input and returns the average of all the item prices.
# Hint - This should determine the total price divided by the number of types of items. This does not account for each item type's quantity.
def get_average_item_price(cart):
    items = cart['items']
    total = 0
    for item in items:
        total = total + item['price']
    return total/len(items)

assert get_average_item_price(shopping_cart) == 2.1420000000000003
print("Exercise 99 is complete.")
# Exercise 100
# Write a function named get_average_spent_per_item that takes in the shopping cart and returns the average of summing each item's quanties times that item's price.
# Hint: You may need to set an initial total price and total total quantity to zero, then sum up and divide that total price by the total quantity
def get_average_spent_per_item(cart):
    items = cart['items']
    total_price = 0
    total_count = 0
    for item in items:
        quantity = item['quantity']
        total_price = total_price + (quantity * item['price'])
        total_count = total_count + quantity
    return total_price/total_count

assert get_average_spent_per_item(shopping_cart) == 1.333529411764706
print("Exercise 100 is complete.")
# Exercise 101
# Write a function named most_spent_on_item that takes in the shopping cart as input and returns the dictionary associated with the item that has the highest price*quantity.
# Be sure to do this as programmatically as possible. 
# Hint: Similarly to how we sometimes begin a function with setting a variable to zero, we need a starting place:
# Hint: Consider creating a variable that is a dictionary with the keys "price" and "quantity" both set to 0. You can then compare each item's price and quantity total to the one from "most"
def most_spent_on_item(cart):
    items = cart['items']
    total_prices = []
    for item in items:
        quantity = item['quantity']
        total_prices.append((quantity * item['price']))
    maxprice = max(total_prices)
    for item in items:
        if maxprice == item['price'] * item['quantity']:
            return item
        
print(most_spent_on_item(shopping_cart))

assert most_spent_on_item(shopping_cart) == {
    "title": "chocolate",
    "price": 0.75,
    "quantity": 9
}
print("Exercise 101 is complete.")
