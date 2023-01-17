# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Excersise 0
doing_python_right_now = True
print(doing_python_right_now)

assert doing_python_right_now == True, "If you see a NameError, it means that the variable is not created and assigned a value. An 'Assertion Error' means that the value of the variable is incorrect." 
print("Exercise 0 is correct") # This line will print if your solution passes the assertion above.
# Excersise 1
on_mars_right_now = False
print(on_mars_right_now)

assert on_mars_right_now == False, "If you see a Name Error, be sure to create the variable and assign it a value."
print("Exercise 1 is correct.")
# Excersise 2
fruits == ["mango", "banana", "guava", "kiwi", "strawberry"]
print(fruits)

assert fruits == ["mango", "banana", "guava", "kiwi", "strawberry"], "If you see an Assert Error, ensure the variable contains all the strings in the provided order"
print("Exercise 2 is correct.")
# Exercise 3
# Create a variable named vegetables and assign it a list of fruits containing the following vegetable names as strings: 
vegetables = ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini"]

assert vegetables == ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini"], "Ensure the variable contains all the strings in the provided order"
print("Exercise 3 is correct.")
# Excersise 4
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(numbers)

assert numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Ensure the variable contains the numbers 1-10 in order."
print("Exercise 4 is correct.")
# Exercise 5
# Given the following assigment of the list of fruits, add "tomato" to the end of the list. 
fruits = ["mango", "banana", "guava", "kiwi", "strawberry"]
fruits.append("tomato")

assert fruits == ["mango", "banana", "guava", "kiwi", "strawberry", "tomato"], "Ensure the variable contains all the strings in the right order"
print("Exercise 5 is correct")
# Exercise 6
# Given the following assignment of the vegetables list, add "tomato" to the end of the list.
vegetables = ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini"]
vegetables.append("tomato")


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
fruits.sort().reverse()

assert fruits == ['tomato', 'strawberry', 'mango', 'kiwi', 'guava', 'banana']
print("Exercise 9 is correct.")
fruits = ["mango", "banana", "guava", "kiwi", "strawberry"]
vegetables = ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini", "tomatoe"]
fruits_and_veggies = (fruits + vegetables)
print(fruits_and_veggies)
print(sorted(fruits_and_veggies))

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
Names = ["George", "John", "Jacob"]
msg = "Hello, " + Names[1] + "!"    
print(msg)


# Example function defintion:
# Write a say_hello function that adds the string "Hello, " to the beginning and "!" to the end of any given input.
def say_hello(name):
    return "Hello, " + name + "!"
print(say_hello("George"))
# Another example function definition:
# This plus_two function takes in a variable and adds 2 to it.
variables = [5,10,15,20,25,30]
plus_2 = variables[0] + 2
print(plus_2)
# Another example function definition:
# This plus_two function takes in a variable and adds 2 to it.
def plus_two(number):
    return number + 2
print(plus_two(7))
# Exercise 11
# Write a function definition for a function named add_one that takes in a number and returns that number plus one.
def add_one(number):
    return number + 1
print(add_one(-5))
# Exercise 12
# Write a function definition named is_positive that takes in a number and returns True or False if that number is positive.
def is_positive(number):
    if number > 0:
        return True
    else:
        return False
print(is_positive(6))
print(is_positive(-6.1))
# Exercise 13
# Write a function definition named is_negative that takes in a number and returns True or False if that number is negative.
def is_negative(number):
    if number < 0:
        return True
    else:
        return False
print(is_negative(-100))
print(is_negative(100))
# Exercise 14
# Write a function definition named is_odd that takes in a number and returns True or False if that number is odd.
def is_odd(number):
    remainder = number % 2
    if remainder > 0:
        return True
    else:
        return False
print(is_odd(6))
# Exercise 15
# Write a function definition named is_even that takes in a number and returns True or False if that number is even.
def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False
print(is_even(100))
# Exercise 16
# Write a function definition named identity that takes in any argument and returns that argument's value. Don't overthink this one!
fruits = ["mango", "banana", "guava", "kiwi", "strawberry"]
vegetables = ["eggplant", "broccoli", "carrot", "cauliflower", "zucchini", "tomatoe"]
meats = ["beef", "pork", "chicken"]
identity = [fruits, vegetables, meats]
print(identity)
# Exercise 17
# Write a function definition named is_positive_odd that takes in a number and returns True or False if the value is both greater than zero and odd
def is_positive_odd(number):
    remainder = number % 2
    if remainder > 0 and number > 0:
        return True
    else:
        return False
print(is_positive_odd(1))
# Exercise 18
# Write a function definition named is_positive_even that takes in a number and returns True or False if the value is both greater than zero and even
def is_positive_even(number):
    if number % 2 == 0 and number > 0:
        return True
    else:
        return False
print(is_positive_even(3))
    
# Exercise 19
# Write a function definition named is_negative_odd that takes in a number and returns True or False if the value is both less than zero and odd.
def is_negative_odd(number):
    remainder = number % 2
    if remainder > 0 and number < 0:
        return True
    else:
        return False
print(is_negative_odd(-3))
    
# Exercise 20
# Write a function definition named is_negative_even that takes in a number and returns True or False if the value is both less than zero and even.
def is_negative_even(number):
    if number % 2 == 0 and number < 0:
        return True
    else:
        return False
print(is_negative_even(-8))
    
# Exercise 21
# Write a function definition named half that takes in a number and returns half the provided number.
def half(number):
    return number/2
print(half(10))
        

# Exercise 22
# Write a function definition named double that takes in a number and returns double the provided number.
def double(number):
    return number *2
print(double(22))
# Exercise 23
# Write a function definition named triple that takes in a number and returns triple the provided number.
def triple(number):
    return number *3
print(triple(3))
# Exercise 24
# Write a function definition named reverse_sign that takes in a number and returns the provided number but with the sign reversed.
def reverse_sign(number):
    return number *-1
print(reverse_sign(-5))
# Exercise 25
# Write a function definition named absolute_value that takes in a number and returns the absolute value of the provided number
def absolute_value(number):
    return abs(number)
print(absolute_value(-10))
# Exercise 26
# Write a function definition named is_multiple_of_three that takes in a number and returns True or False if the number is evenly divisible by 3.
def is_multiple_of_three(number):
    if number % 3 == 0 and number != 0:
        return True
    else:
        return False
print(is_multiple_of_three(9))

    
    
# Exercise 27
# Write a function definition named is_multiple_of_five that takes in a number and returns True or False if the number is evenly divisible by 5.
def is_multiple_of_five(number):
    if number % 5 == 0 and number !=0:
        return True
    else:
        return False
print(is_multiple_of_five(10))
    
# Exercise 28
# Write a function definition named is_multiple_of_both_three_and_five that takes in a number and returns True or False if the number is evenly divisible by both 3 and 5.
def is_multiple_of_both_three_and_five(number):
    if number % 3 == 0 and number % 5 ==0 and number!=0:
        return True
    else:
        return False
print(is_multiple_of_both_three_and_five(0))

# Exercise 29
# Write a function definition named square that takes in a number and returns the number times itself.
def square(number):
    return number * number
print(square(5))

# Exercise 30
# Write a function definition named add that takes in two numbers and returns the sum.
def add(number1,number2):
    return number1 + number2
print(add(5,2))
# Exercise 31
# Write a function definition named cube that takes in a number and returns the number times itself, times itself.
def cube(number):
    return number ** 3
print(cube(4))
# Exercise 32
# Write a function definition named square_root that takes in a number and returns the square root of the provided number
def square_root(number):
    return number ** 0.5
print(square_root(49))
# Exercise 33
# Write a function definition named subtract that takes in two numbers and returns the first minus the second argument.
def subtract(number1,number2):
    return number1 - number2
print(subtract(8,3))

# Exercise 34
# Write a function definition named multiply that takes in two numbers and returns the first times the second argument.
def multiply(number1,number2):
    return number1 * number2
print(multiply(5,1))
    
# Exercise 35
# Write a function definition named divide that takes in two numbers and returns the first argument divided by the second argument.
def divide(number1,number2):
    return number1 / number2
print(divide(8,4))
# Exercise 36
# Write a function definition named quotient that takes in two numbers and returns only the quotient from dividing the first argument by the second argument.
def quotent(number1, number2):
    return number1 / number2
print(quotent(27,9))

# Exercise 37
# Write a function definition named remainder that takes in two numbers and returns the remainder of first argument divided by the second argument.
def remainder(number1,number2):
    return number1 % number2
print(remainder(12,4))
# Exercise 38
# Write a function definition named sum_of_squares that takes in two numbers, squares each number, then returns the sum of both squares.
def sum_of_squares(number1,number2):
    return number1 **2 + number2 **2
print(sum_of_squares(2,4))

# Exercise 39
# Write a function definition named times_two_plus_three that takes in a number, multiplies it by two, adds 3 and returns the result.
def times_two_plus_three(number):
    return number * 2 + 3
print(times_two_plus_three(2))
# Exercise 40
# Write a function definition named area_of_rectangle that takes in two numbers and returns the product.
def area_of_rectangle(number1,number2):
    return number1 * number2
print(area_of_rectangle(5,2))
import math
# Exercise 41
# Write a function definition named area_of_circle that takes in a number representing a circle's radius and returns the area of the circl
def area_of_circle(radius):
    return radius **2 * 3.1415926535
print(area_of_circle(5))
    
    
# Exercise 42
# Write a function definition named circumference that takes in a number representing a circle's radius and returns the circumference.
def circumference(radius):
    return 3.1415926535 * 2 *  radius
print(circumference(5))
    
# Exercise 43
# Write a function definition named is_vowel that takes in value and returns True if the value is a, e, i, o, u in upper or lower case.
def is_vowel(letters):
    if letters =="a" or letters =="e" or letters =="i" or letters =="o" or letters =="u" or letters =="A" or letters =="E" or letters =="I" or letters =="O" or letters =="U":
        return True
    else:
        return False
print(is_vowel("E"))
    
    

# Exercise 44
# Write a function definition named has_vowels that takes in value and returns True if the string contains any vowels.
def has_vowel(letters):
    for char in letters:
        if is_vowel(char):
            return True
    return False
print(has_vowel("ytre"))
# Exercise 45
# Write a function definition named count_vowels that takes in value and returns the count of the nubmer of vowels in a sequence.
def count_vowels(words):
    list=[]
    for char in words:
        if has_vowel(char):
            list.append(char)
    return len(list)
print(count_vowels("I got it"))

# Exercise 46
# Write a function definition named remove_vowels that takes in string and returns the string without any vowels
def remove_vowels(word):
    vowels = ('a','e','i','o','u')
    for char in word:
        if char in vowels:
            word=word.replace(char, "")
    return word
print(remove_vowels("apple"))
    
# Exercise 47
# Write a function definition named starts_with_vowel that takes in string and True if the string starts with a vowel
def starts_with_vowel(word):
    split = word.split()
    for char in split:
        if char[0] in ['a','e','i','o','u']:
            return True
        else:
            return False
print(starts_with_vowel("eat"))
# Exercise 48
# Write a function definition named ends_with_vowel that takes in string and True if the string ends with a vowel
def ends_with_vowel(word):
    split = word.split()
    for char in split:
        if char[-1] in ['a','e','i','o','u']:
            return True
        else:
            return False
print(ends_with_vowel("eat"))
# Exercise 49
# Write a function definition named starts_and_ends_with_vowel that takes in string and returns True if the string starts and ends with a vowel
def starts_and_ends_with_vowel(word):
    split = word.split()
    for char in split:
        if char[0] in ['a','e','i','o','u'] and char[-1] in ['a','e','i','o','u']:
            return True
        else:
            return False
print(starts_and_ends_with_vowel("eate"))
# Exercise 50
# Write a function definition named first that takes in sequence and returns the first value of that sequence.
def first(word):
        return word[0]
print(first(["The","dog","big"]))
print(first(["1",'2','3']))
print(first("123"))
# Exercise 51
# Write a function definition named second that takes in sequence and returns the second value of that sequence.
def second(word):
        return word[1]
print(second(["The","dog","big"]))
print(second(["1",'2','3']))
print(second("123"))
# Exercise 52
# Write a function definition named third that takes in sequence and returns the third value of that sequence.
def third(word):
        return word[2]
print(third(["The","dog","big"]))
print(third(["1",'2','3']))
print(third("123"))
# Exercise 53
# Write a function definition named forth that takes in sequence and returns the forth value of that sequence.
def forth(word):
    return word[3]
print(forth(['doggy','trainer','funny',"minus"]))
print(forth('1234'))
# Exercise 54
# Write a function definition named last that takes in sequence and returns the last value of that sequence.
def last(word):
    return word[-1]
print(last(['doggy','trainer','funny',"minus"]))
print(last('123'))
# Exercise 55
# Write a function definition named second_to_last that takes in sequence and returns the second to last value of that sequence.
def second_to_last(word):
    return word[-2]
print(second_to_last(['doggy','trainer','funny',"minus"]))
print(second_to_last('123'))
# Exercise 56
# Write a function definition named third_to_last that takes in sequence and returns the third to last value of that sequence.
def third_to_last(word):
    return word[-3]
print(third_to_last(['doggy','trainer','funny',"minus"]))
print(third_to_last('123'))
# Exercise 57
# Write a function definition named first_and_second that takes in sequence and returns the first and second value of that sequence as a list
def first_and_second(word):
    words =  [word[i] for i in (0, 1) ]
    return words
print(first_and_second(['doggy','trainer','funny',"minus"]))
print(first_and_second('123'))

# Exercise 58
# Write a function definition named first_and_last that takes in sequence and returns the first and last value of that sequence as a list
def first_and_last(word):
    words =  [word[i] for i in (0, -1) ]
    return words
print(first_and_last(['doggy','trainer','funny',"minus"]))
print(first_and_last('123'))
# Exercise 59
# Write a function definition named first_to_last that takes in sequence and returns the sequence with the first value moved to the end of the sequence.
def first_to_last(word):
    word.append(word[0])
    word.remove(word[0])
    return word
print(first_to_last(['doggy','trainer','funny',"minus"]))
print(first_to_last(['1','2','3','4']))
# Exercise 60
# Write a function definition named sum_all that takes in sequence of numbers and returns all the numbers added together.
def sum_all(numbers):
    for num in numbers:
        return sum(numbers)
print(sum_all([3,2,3,4]))
# Exercise 61
# Write a function definition named mean that takes in sequence of numbers and returns the average value
def mean(numbers):
    for num in numbers:
        return sum(numbers)/len(numbers)
print(mean([5,5,5,5]))
# Exercise 62
# Write a function definition named median that takes in sequence of numbers and returns the average value
import statistics
def median(numbers):
    for num in numbers:
        return statistics.median(numbers)
print(median([1,2,3,4,5]))
# Exercise 63
# Write a function definition named mode that takes in sequence of numbers and returns the most commonly occuring value
import statistics
def mode(numbers):
    for num in numbers:
        return statistics.mode(numbers)
print(mode([1,2,3,4,5,1]))

# Exercise 64
# Write a function definition named product_of_all that takes in sequence of numbers and returns the product of multiplying all the numbers together
def product_of_all(number):
    total = 1
    for x in number:
        total *= x
    return total
print(product_of_all((1,2,3,-4)))

# Run this cell in order to use the following list of numbers for the next exercises
numbers = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5] 
# Exercise 65
# Write a function definition named get_highest_number that takes in sequence of numbers and returns the largest number.
def get_highest_number(number):
    for num in number:
        return max(number)
print(get_highest_number([1,22,3,55,10]))
# Exercise 66
# Write a function definition named get_smallest_number that takes in sequence of numbers and returns the smallest number.
def get_the_smallest_number(number):
    for num in number:
        return min(number)
print(get_the_smallest_number([1,22,3,55,10]))
# Exercise 67
# Write a function definition named only_odd_numbers that takes in sequence of numbers and returns the odd numbers in a list.
def only_odd_numbers(number):
    list=[]
    for num in number:
        if num % 2 != 0:
            list.append(num)
    return list
print(only_odd_numbers([1,2,3,4,5,6]))
# Exercise 68
# Write a function definition named only_even_numbers that takes in sequence of numbers and returns the even numbers in a list.
def only_even_numbers(number):
    list=[]
    for num in number:
        if num % 2 ==0:
            list.append(num)
    return list
print(only_even_numbers([1,2,3,4,5,6]))
# Exercise 69
# Write a function definition named only_positive_numbers that takes in sequence of numbers and returns the positive numbers in a list.
def only_positive_numbers(number):
    list=[]
    for num in number:
        if num > 0:
            list.append(num)
    return list
print(only_positive_numbers([-1,-3,3,1]))
# Exercise 70
# Write a function definition named only_negative_numbers that takes in sequence of numbers and returns the negative numbers in a list.
def only_negative_numbers(number):
    list=[]
    for num in number:
        if num <0:
            list.append(num)
    return list
print(only_negative_numbers([-1,-3,1,3]))
# Exercise 71
# Write a function definition named has_evens that takes in sequence of numbers and returns True if there are any even numbers in the sequence
def has_evens(number):
    for num in number:
        if num % 2 == 0:
            return True
    else:
        return False
print(has_evens([1,5,3,4]))
# Exercise 72
# Write a function definition named count_evens that takes in sequence of numbers and returns the number of even numbers
def count_evens(number):
    list=[]
    for num in number:
        if num % 2 ==0:
            list.append(num)
    return len(list)
print(count_evens([1,2,8,4]))
# Exercise 73
# Write a function definition named has_odds that takes in sequence of numbers and returns True if there are any odd numbers in the sequence
def has_odds(number):
    for num in number:
        if num % 2 != 0:
            return True
    else:
        return False
print(has_odds([2,4,6,3]))
# Exercise 74
# Write a function definition named count_odds that takes in sequence of numbers and returns True if there are any odd numbers in the sequence
def count_odds(number):
    list=[]
    for num in number:
        if num %2 != 0:
            list.append(num)
    return len(list)
print(count_odds([1,2,3,5,7]))
# Exercise 75
# Write a function definition named count_negatives that takes in sequence of numbers and returns a count of the number of negative numbers
def count_negatives(number):
    list=[]
    for num in number:
        if num <0:
            list.append(num)
    return len(list)
print(count_negatives([-1,-3,1,3,-5]))
# Exercise 76
# Write a function definition named count_positives that takes in sequence of numbers and returns a count of the number of positive numbers
def count_positives(number):
    list=[]
    for num in number:
        if num %2 ==0:
            list.append(num)
    return len(list)
print(count_positives([1,2,4,5,6]))
# Exercise 77
# Write a function definition named only_positive_evens that takes in sequence of numbers and returns a list containing all the positive evens from the sequence
def only_positive_evens(number):
    list=[]
    for num in number:
        if num % 2 == 0 and num > 0:
            list.append(num)
    return list
print(only_positive_evens([-1,1,2,-4,4]))
# Exercise 78
# Write a function definition named only_positive_odds that takes in sequence of numbers and returns a list containing all the positive odd numbers from the sequence
def only_positive_odds(number):
    list=[]
    for num in number:
        if num % 2 != 0 and num > 0:
            list.append(num)
    return list
print(only_positive_odds([-1,1,2,-4,4,3]))
# Exercise 79
# Write a function definition named only_negative_evens that takes in sequence of numbers and returns a list containing all the negative even numbers from the sequence
def only_negative_evens(number):
    list=[]
    for num in number:
        if num < 0 and num % 2 == 0 :
            list.append(num)
    return list
print(only_negative_evens([-1,1,2,-4,4,3]))
# Exercise 80
# Write a function definition named only_negative_odds that takes in sequence of numbers and returns a list containing all the negative odd numbers from the sequence
def only_negative_odds(number):
    list=[]
    for num in number:
        if num < 0 and num % 2 != 0:
            list.append(num)
    return list
print(only_negative_odds([-1,1,2,-4,4,-3]))
# Exercise 81
# Write a function definition named shortest_string that takes in a list of strings and returns the shortest string in the list.
def shortest_string(word):
    return min(word, key=len)
print(shortest_string(['better','doggy','cat','mo']))
# Exercise 82
# Write a function definition named longest_string that takes in sequence of strings and returns the longest string in the list.
def longest_string(word):
    return max(word, key=len)
print(longest_string(['better','doggy','cat','Santa Clause']))
# Exercise 83
# Write a function definition named get_unique_values that takes in a list and returns a set with only the unique values from that list.
def get_unique_values(word):
    list=[]
    for char in word:
        if not char in list:
            list.append(char)
    return list
print(get_unique_values('dooggey'))
    
# Exercise 84
# Write a function definition named get_unique_values_from_two_lists that takes two lists and returns a single set with only the unique values
def get_unique_values_from_two_lists(list0, list1):
    return set(list0 + list1)
print(get_unique_values_from_two_lists('dog','fog'))
# Exercise 85
# Write a function definition named get_values_in_common that takes two lists and returns a single set with the values that each list has in common
def get_values_in_common(list0,list1):
    list=[]
    for char in list0 and list1:
        if char in list0 and list1:
            list.append(char)
    return list
print(get_values_in_common('dog','fog'))
# Exercise 86
# Write a function definition named get_values_not_in_common that takes two lists and returns a single set with the values that each list does not have in common
def get_values_not_in_common(list0,list1):
    return (set(list0)^set(list1))
print(get_values_not_in_common('dog','fog'))
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
def get_paper_title(title):
    return (tukey_paper.get("title"), thomas_paper.get("title"))
print(get_paper_title("title"))


# Exercise 88
# Write a function named get_year_published that takes in a dictionary and returns the value behind the "year_published" key.
def get_year_published(year):
    return (tukey_paper.get('year_published'), thomas_paper.get("year_published"))
print(get_year_published("year_published"))
# Run this code to create data for the next two questions
book = {
    "title": "Genetic Algorithms and Machine Learning for Programmers",
    "price": 36.99,
    "author": "Frances Buontempo"
}
# Exercise 89
# Write a function named get_price that takes in a dictionary and returns the price
def get_price(price):
    return (book.get("price"))
print(get_price("price"))
# Exercise 90
# Write a function named get_book_author that takes in a dictionary (the above declared book variable) and returns the author's name
def get_book_author(author):
    return (book.get("author"))
print(get_book_author("author"))
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
def get_number_of_books(l):
    return sum(isinstance(i, dict) for i in l) 
print(get_number_of_books(books))
    
# Exercise 92
# Write a function named total_of_book_prices that takes in a list of dictionaries and returns the sum total of all the book prices added together
def total_of_book_prices(prices):
    return sum([i['price'] for i in prices])
print(total_of_book_prices(books))

# Exercise 93
# Write a function named get_average_book_price that takes in a list of dictionaries and returns the average book price.
def get_average_book_price(prices):
    return sum([i['price'] for i in prices])/len(prices)
print(get_average_book_price(books))
# Exercise 94
# Write a function called highest_priced_book that takes in the above defined list of dictionaries "books" and returns the dictionary containing the title, price, and author of the book with the highest priced book.
# Hint: Much like sometimes start functions with a variable set to zero, you may want to create a dictionary with the price set to zero to compare to each dictionary's price in the list
def highest_priced_book(prices):
    return(max((i for i in prices), key = lambda k: k['price']))
print(highest_priced_book(books))
# Exercise 95
# Write a function called lowest_priced_book that takes in the above defined list of dictionaries "books" and returns the dictionary containing the title, price, and author of the book with the lowest priced book.
# Hint: Much like sometimes start functions with a variable set to zero or float('inf'), you may want to create a dictionary with the price set to float('inf') to compare to each dictionary in the list
def lowest_priced_book(prices):
    return(min((i for i in prices), key = lambda k: k['price']))
print(lowest_priced_book(books))
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
def get_tax_rate(prices):
    for key, value in shopping_cart.items():
        return value
print(get_tax_rate('tax'))

# Exercise 97
# Write a function named number_of_item_types that takes in the shopping cart as input and returns the number of unique item types in the shopping cart. 
# We're not yet using the quantity of each item, but rather focusing on determining how many different types of items are in the cart.
def number_of_item_types(price):
    for items in shopping_cart:
        return len(shopping_cart.items())
print(number_of_item_types("items"))
    
# Exercise 97
# Write a function named number_of_item_types that takes in the shopping cart as input and returns the number of unique item types in the shopping cart. 
# We're not yet using the quantity of each item, but rather focusing on determining how many different types of items are in the cart.
def number_of_item_types(price):
    count = 0
    for x in shopping_cart:
        if isinstance(shopping_cart[x], list): 
            count += len(shopping_cart[x])
    return len(list)
print(number_of_item_types("items"))
# Exercise 98
# Write a function named total_number_of_items that takes in the shopping cart as input and returns the total number all item quantities.
# This should return the sum of all of the quantities from each item type
# Exercise 99
# Write a function named get_average_item_price that takes in the shopping cart as an input and returns the average of all the item prices.
# Hint - This should determine the total price divided by the number of types of items. This does not account for each item type's quantity.
# Exercise 100
# Write a function named get_average_spent_per_item that takes in the shopping cart and returns the average of summing each item's quanties times that item's price.
# Hint: You may need to set an initial total price and total total quantity to zero, then sum up and divide that total price by the total quantity
# Exercise 101
# Write a function named most_spent_on_item that takes in the shopping cart as input and returns the dictionary associated with the item that has the highest price*quantity.
# Be sure to do this as programmatically as possible. 
# Hint: Similarly to how we sometimes begin a function with setting a variable to zero, we need a starting place:
# Hint: Consider creating a variable that is a dictionary with the keys "price" and "quantity" both set to 0. You can then compare each item's price and quantity total to the one from "most"