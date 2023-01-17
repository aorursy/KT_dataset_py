print("This is my first line of code in python")
print("Hello\nThis is in the next line")
a = 6
b = 7
print(a + b)  

print(a - b) 
print(a * b) 
print(a / b)
print(a // b)
print(a % b) 
print(a ** b)
#BEDMAS
print(4 * 5 - 9 + 6 / 7)
inc = 9
inc += 1
print(inc)
inc -= 1
print(inc)
#Logical
print(3 < 4)
print(True and False) 
print(True or False)
print(a!=b) 
print(True and not False)
small = "i am upper cased"
print(small.upper())

large = "I AM LOWER CASED"
print(large.lower())

some_sentence = "There is a space at the end    "
print(some_sentence)
print(some_sentence.rstrip())

increment = '4%'
print(increment.rstrip('%')) 
start = "   There is space at the start"
print(start)
print(start.lstrip())
spaces = "   Trim whitespaces  "
print(spaces)
print(spaces.strip())
num_with_chars = '*444#'
print(num_with_chars.rstrip('#').lstrip('*'))
val = "2 apples"
no_of_apples = val[0] 
print('number of apples is', no_of_apples)

what = val[2:]
print(what)
#slicing by specifying start and end index
print(val[2:5])

batch = "5 oranges 3 monkeys n"
fruits = batch[ :9]
print(fruits)
print(batch[ :-2])
animals = batch[10:-2]
print(animals)
nums = '123456789'
even_nums = nums[1::2]
print(even_nums)
odd_nums = nums[0::2]
print(odd_nums)
odd_nums_again = nums[ :: 2]

first_name = "Monty"
last_name = 'Python'
name = first_name + " " + last_name
print(name)
age = 30
my_age = ("I am "+ str(age) + " years old")
My_age = "I am {0} years old".format(age) 
A = "Data"
B = "Analysis"
C = "Pandas"

print("{0} {1} using {2}".format(A,B,C))

#concat and update
name += ' ' + my_age
print(name)