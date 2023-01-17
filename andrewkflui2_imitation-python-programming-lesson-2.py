amount = int(input('Enter an amount of money (in dollars): '))

notesCount = amount // 500  # calculate number of $500 bank notes
remainder = amount % 500   # find the remaining

print('Converted to {} $500 notes and the remaining is ${}'.format(notesCount, remainder))
# Edit your solution here
amount = int(input('Enter an amount of money (in dollars): '))

print('Converted to {} $500 notes and the remaining is ${}'.format(amount // 500, amount % 500))
a = 25
b = 10

# Arithmetic Operators

print('a + b = {}'.format(a + b))  # 35
print('a - b = {}'.format(a - b))
print('a * b = {}'.format(a * b))
print('a / b = {}'.format(a / b))

print('a // b = {}'.format(a // b))
print('a % b = {}'.format(a % b))

# Relational Operators

print('a < b = {}'.format(a < b))
print('a > b = {}'.format(a > b))
print('a <= b = {}'.format(a <= b))
print('a >= b = {}'.format(a >= b))
print('a == b = {}'.format(a == b))
print('a != b = {}'.format(a != b))

# Logical Operators

print('a == 25 and b == 10 {}'.format(a == 25 and b == 10))
print('a == 25 or b == 10 {}'.format(a == 25 or b == 10))
print('a == 25 and b == 0 {}'.format(a == 25 and b == 0))
print('a == 25 or b == 0 {}'.format(a == 25 or b == 0))
numberOfCans = int(input('Enter the number of cans purchased: '))

freeCan = numberOfCans // 5

if numberOfCans >= 30:
    freeCan = freeCan + 3  # 3 additional cans are given if more than 30 cans
    
print('The customer should receive extra {} cans free'.format(freeCan))
# Edit your solution here
a = 25
b = 10
c = 0

if True:
    print('One')

if False:
    print('Two')      # Skipped
    
if a > b:
    print('Three')
    print('Four')
    print('Five')

if a:
    print('Six')

if c:
    print('Seven')     # Skipped

if c == 0:
    print('Eight')

# Make changes here

a = 25
b = 10
c = 0

if True:
    print('One')

if False:
    print('Two')      # Skipped
    
if a > b:
    print('Three')
    print('Four')
    print('Five')

if a:
    print('Six')

if c:
    print('Seven')     # Skipped

if c == 0:
    print('Eight')
numberOfCans = int(input('Enter the number of cans purchased: '))

freeCan = numberOfCans // 5

if numberOfCans >= 30 and numberOfCans <= 60:
    freeCan = freeCan + 3  # 3 additional cans are given if between 30 and 60 purchases
if numberOfCans > 60:
    freeCan = freeCan + 5  # 3 additional cans are given if more than 60 purchases
    
print('The customer should receive extra {} cans free'.format(freeCan))
weight = float(input('Enter weight in kg: '))

height = float(input('Enter height in m: '))

bmi = weight / (height * height)

print('The BMI is {}'.format(bmi))
# Enter your solution here
# Exercise 1

amount = int(input('Enter number of participants: '))

table = amount // 12  # calculate number of $500 bank notes
remainder = amount % 12   # find the remaining

print('Requires {} tables and the remaining participants is {}'.format(table, remainder))
# Exercise 1

amount = int(input('Enter number of participants: '))

table = amount // 12  # calculate number of $500 bank notes
remainder = amount % 12   # find the remaining

if remainder > 3:
    table = table + 1

print('Requires {} tables'.format(table))  # note that there should be no remainder now
# Exercise

# Just change the condition for the if structure to be True

a = 25
b = 10
c = 0

if True:
    print('One')

if True:
    print('Two')     
    
if a > b:
    print('Three')
    print('Four')
    print('Five')

if a:
    print('Six')

if True:
    print('Seven')     # Skipped

if c == 0:
    print('Eight')

weight = float(input('Enter weight in kg: '))

height = float(input('Enter height in m: '))

bmi = weight / (height * height)

print('The BMI is {}'.format(bmi))

if bmi < 19:
    print('Underweight')
if bmi >= 19 and bmi <= 22:
    print('Just right')
if bmi > 22 and bmi <= 25:
    print('Overweight')
if bmi > 25:
    print('Health at risk')