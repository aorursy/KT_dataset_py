name1 = "John"
name2 = "Smith"
full_name = name1 + " " + name2
print(full_name)

# .format()
name = "John"
age = 29
print("Hello {}, you are {}".format(name, age))
# f strings
name = "Harry"
print(f"Hello {name}")
# .title()
name = "john smith"
print(name.title())
# .replace()
words = "Hello, there!"
print(words.replace("!", "?"))
# .find()
sentence = "Go ahead and run that cell. You’ll notice that we got an output of 5. Find returns the starting index position of the match."
sentence.find("we")
# .strip()
name = " john "
print(name.strip())
# .split()
sentence = "Go ahead and run that cell. You’ll notice that we got an output of 5. Find returns the starting index position of the match."
sentence.split(" ")
# create a product and price for three items
p1_name, p1_price = "Books", 49.95
p2_name, p2_price = "Computer", 579.99
p3_name, p3_price = "Monitor", 124.89

# create a company name and information
company_name = "coding temple, inc."
company_address = "283 Franklin St."
company_city = "Boston, MA"

# declare ending message
message = "Thanks for shopping with us today!"
# create a top border
print( "*" * 50 )

# print company information first, using format
print("\t\t{}".format(company_name.title()))
print("\t\t{}".format(company_address))
print("\t\t{}".format(company_city))

# print a line between sections
print( "=" * 50 )

# print out header for section of items
print("\tProduct Name\tProduct Price")

# create a print statement for each product
print("\t{}\t\t${}".format(p1_name, p1_price))
print("\t{}\t${}".format(p2_name, p2_price))
print("\t{}\t\t${}".format(p3_name, p3_price))

# print a line between sections
print('=' * 50)

# print out header for section of total
print("\t\t\tTotal")

# calculate total price and print out
total = p1_price + p2_price + p3_price
print("\t\t\t${}".format(total))

# print a line between sections
print( "=" * 50)

# output thank you message
print("\n\t{}\n".format(message))

# create a bottom border
print( "*" * 50 )
print(input("What is your name?"))
ans = input("What is your name?")
print("Hello {}!".format(ans))
type(ans)
#Checking the Type
num = 5
print(type(num))
# Converting Data Types
num = "9"
num = int(num)
print(type(num))
ans = input("Type a number to add: ")
print("initial type: ", type(ans))
result = 100 + int(ans)
print("100 + {} = {}".format(ans, result))
try:
    ans = float(input("Type a number to add: "))
    print("100 + {} = {}".format(ans, 100+ans))
    print("The program did not break!")
except:
    print("You did not put in a valid number!")
# without try/except print statement would not get hit if error occurs

try:
    ans = float(input("Type a number to add: "))
    print("100 + {} = {}".format(ans, 100+ans))
    print("The program did not break")
except:
    print("You did not put in a valid number!")
# without try/except print statement would not get hit if error occurs
x, y = 5, 10
if x < y: 
    print("x is less than y")
x, y, z = 5, 10, 5
if x < y and x == y:
    print("both statements were true")
else:
    print("false")
#membership operator "in"
word = "Baseball"
if 'b' in word:
    print("{} contain the character b".format(word))
#membership operator "not in"
word = "Baseball"
if 'z' in word:
    print("{} contain the character z".format(word))
else:
    print("{} doesn't contain the character z".format(word))
# using the elif conditional statement
x, y = 5, 10
if x > y:
    print("x is greater")
elif x < y:
    print("x is less")
# checking more than one elif conditional statement
x, y = 5, 10
if x > y:
    print("x is greater")
elif (x + 10) < y:
    print("x is less")
elif (x + 5) == y:
    print("equal")
# writing multiple conditionals within each other - multiple block levels
x, y, z = 5, 10, 5
if x > y:
    print("greater")
elif x <= y:
    if x == z:
        print("x is equal to z")
    elif x != z:
        print("x is not equal to z")
# using an else statement
name = "John"
if name == "Jacob":
    print("Hello Jacob!")
else:
    print("Hello {}!".format(name))
# writing a full conditional statement with if, elif, else
name = "sJohn"
if name[0] == "A":
    print("Name starts with an A")
elif name[0] == "B":
    print("Name starts with a B")
elif name[0] == "J":
    print("Name starts with a J")
else: # covers all other possibilities
    print( "Name starts with a {}".format( name[0] ) )
# Step 1: Ask User for Calculation to Be Performed
operation = input("Would you like to add/substract/multiply/divide? ").lower()
print("You chose to {}.".format(operation))
# Step 2: Ask for Numbers, Alert Order Matters
if operation == "substract" or operation == "divide":
    print("You chose to {}.".format(operation))
    print("Please keep in mind that the order of your numbers matter.")
num1 = input("What is the first number?")
num2 = input("What is the second number?")
print("First Number: {}".format(num1))
print("Second Nunber: {}".format(num2))
# Step 3: Set Up Try/Except for Mathematical Operation
try:
    # step 3a: immediately try to convert numbers input to floats
    num1, num2 = float(num1), float(num2)
    # step 3b: perform operation and print result
    if operation == "add":
        result = num1 + num2
        print("{} + {} = {}".format(num1, num2, result))
    elif operation == "substract":
        result = num1 - num2
        print("{} - {} = {}".format(num1, num2, result))
    elif operation == "multiply":
        result = num1 * num2
        print("{} * {} = {}".format(num1, num2, result))
    elif operation == "divide":
        result = num1 / num2
        print("{} / {} = {}".format(num1, num2, result))
    else:
        # else will be hit if they didn't chose an option correctly
        print("Sorry, but '{}' is not an option.".format(operation))
except:
    # steb 3c: print error
    print("Error: Improper numbers used. Please try again")
# declaring a list of numbers
nums = [5, 10, 15.2, 20]
print(nums)
# accessing elements within a list
print( nums[1] ) # will output the value at index 1 = 10
num = nums[2] # saves index value 2 into num
print(num) # prints value assigned to num
# declaring a list of mixed data types
num = 4.3
data = [num, "word", True] # the power of data collection
print(data)
# understanding lists within lists
data = [5, "book", [ 34, "hello" ], True] # lists can hold any type
print(data)
print( data[2] )
# using double bracket notation to access lists within lists
print( data[2][0] ) # will output 34
inner_list = data[2] # inner list will equal [34, 'hello']
print( inner_list[1] ) # will output 'hello'
# changing values in a list through index
data = [5, 10, 15, 20]
print(data)
data[0] = 100 # change the value at index 0 - (5 to 100)
print(data)
a = [5, 10]
print(id(a))
# Changing the value at a specific index will change the value for both lists. Let’s see an example:
# understanding how lists are stored
a = [5, 10]
b = a
print( "a: {}\t b: {}".format(a, b) )
print( "Location a[0]: {}\t Location b[0]: {}".format( id(a[0]), id(b[0]) ) )
a[0] = 20 # re-declaring the value of a[0] also changes b[0]
print( "a: {}\t b: {}".format(a, b) )
# using [:] to copy a list
data = [5, 10, 15, 20]
data_copy = data[:]
data[0] = 50
print("data: {}\t data_copy: {}".format(data, data_copy))
# writing your first for loop using range
for num in range(5):
    print( "Value: {}".format(num) )
# providing the start, stop, and step for the range function
for num in range(2, 10, 2):
    print( "Value: {}".format(num) ) # will print all evens between 2and 10
# printing all characters in a name using the 'in' keyword
name = "John Smith"
for letter in name:
    print( "Value: {}".format(letter) )
# using the continue statement within a foor loop
for num in range(5):
    if num == 3:
        continue
    print(num)
# breaking out of a loop using the 'break' keyword
for num in range(5):
    if num == 3:
        break
    print(num)
# setting a placeholder using the 'pass' keyword
for i in range(5):
    # TODO: add code to print number
    pass
# writing your first while loop
health = 10
while health > 0:
    print(health)
    health -= 1 # forgetting this line will result in infinite loop
# This is an example of infinite loops
# game_over = False
# while not game_over:
#     print(game_over)
# using two or more loops together is called a nested loop
for i in range(2): # outside loop
    for j in range(3): # inside loop
        print( i, j )
# checking the number of items within a list
nums = [5, 10, 15]
length = len(nums) # len() returns an integer
print(length)
# accessing specific items of a list with slices
print( nums[ 1 : 3 ] ) # will output items in index 1 and 2
print( nums[ : 2 ] ) # will output items in index 0 and 1
print( nums[ : : 2 ] ) # will print every other index - 0, 2, 4, etc.
print( nums[ -2 : ] ) # will output the last two items in list
# .append()
# adding an item to the back of a list using append
nums = [10, 20]
nums.append(5)
print(nums) # outputs [10, 20, 5]
# .insert( )
# adding a value to the beginning of the list
words = [ "ball", "base" ]
nums.insert(0, "glove") # first number is the index, second is the value
nums
# .pop()
# using pop to remove items and saving to a variable to use later
items = [5, "ball", True]
items.pop( ) # by default removes the last item
removed_item = items.pop(0) # removes 5 and saves it into the variable
print(removed_item, "\n", items)
# .remove()
# using the remove method with a try and except
sports = [ "baseball", "soccer", "football", "hockey" ]
try:
    sports.remove("soccer")
except:
    print("That item does not exist in the list")
print(sports)
# using min, max, and sum
nums = [5, 3, 9]
print( min(nums) ) # will find the lowest number in the list
print( max(nums) ) # will find the highest number in the list
print( sum(nums) ) # will add all numbers in the list and return the sum
# sorted
# using sorted on lists for numerical and alphabetical data
nums = [5, 8, 0, 2]
sorted_nums = sorted(nums) # save to a new variable to use later
print(nums, sorted_nums) # the original list is in tact
# sort
# sorting a list with .sort() in-place
nums = [5, 0, 8, 3]
nums.sort( ) # alters the original variable directly
print(nums)
# using conditional statements on a list
names = [ "Jack", "Robert", "Mary" ]
if "Mary" in names:
    print("found") # will run since Mary is in the list
if "Jimmy" not in names:
    print("not found") # will run since Jimmy is not in the list
# using a for loop to print all items in a list
sports = [ "Baseball", "Hockey", "Football", "Basketball" ]
for sport in sports:
    print(sport)
# using the while loop to remove a certain value
names = [ "Bob", "Jack", "Rob", "Bob", "Robert" ]
while "Bob" in names:
    names.remove("Bob") # removes all instances of 'Bob'
print(names)
page 






a = 100
def amount():
    a = 79
    return a
print(amount())
sentence = "I am global"
def printing():
#     sentence = "I am local"
    
    def execution():
        print("result is " + sentence)
    execution()
    
printing()
sentence = "I am global"
def printing():
    sentence = "I am local"
    
    def execution():
        print("result is " + sentence)
    execution()
    
printing()
sentence = "I am global"
def printing():
    sentence = "I am local"
    
    def execution():
        sentence = "I am very local"
        print("result is " + sentence)
    execution()
    
printing()
number = 99
def write(number):
    print(f'number = {number}')
    
#     redefine the variable number
    number = 300
    print(f'the number is changed into {number}')

write(number)
print(number)
number = 99
def write():
    global number

#     redefine the variable number
    number = "CHANGED NUMBER"
    print(f'the number is changed into {number}')

write()
print(number)
# recommended way to change the global variable
number = 99
def write(number):
#     redefine the variable number
    number = "CHANGED NUMBER"
    print(f'the number is changed into {number}')
    return number

number = write(number)
print(number)
def multiply10(num):
    return num*10
multiply10(20)
number = [3, 6, 9]
multiply10(number) #not like this
for item in map(multiply10, number):
    print(item)
result = list(map(multiply10, number))
result
def words_number(word) :
    if len(word) % 2 == 0:
        return 'It is even'
    else: 
        return 'It is odd'
words_number('humiliate')
word_list = ['hurry', 'even', 'moment']
result = list(map(words_number, word_list))
result
def check_even(num):
    return num%2 == 0
check_even(10)
number = [1, 2, 3, 4, 5, 6, 7, 8]
result = list(filter(check_even, number))
result
# conventional function
def multiply10(num):
    return num*10
# lambda expression
lambda num: num * 10
my_function = lambda num: num * 10
my_function(10)
# with list
list_result = list(map(lambda x: x*10, number))
list_result
even_number = list(filter(lambda num: num%2 == 0, number))
even_number

print(type([]))
print(type({}))
print(type('it is string'))
print(type(12))
print(type(13.98))
print(type(()))
print(type(True))
print(type(None))
# Python 3.x
class NamaClass:
    pass

# Python 2.x
class NamaClass2():
    pass
var1 = NamaClass()
type(var1)
# class Kulkas:
    
#     def __init__(self, merek, harga):
#         self.merek = merek
#         self.harga = harga
        
class Kulkas:
    
    def __init__(self, merek, harga):
        self.brand = merek
        self.price = harga
        self.guarantee = 3
        self.power = "electricity"
# item1 = Kulkas(merek="Samsung", harga=500)
item1 = Kulkas("Samsung", 500) # --> self in class represent the assigned variable
item2 = Kulkas("Toshiba", 600)
item1.brand, item1.price, item1.guarantee
# those are similar to self.brand, self.price
item2.brand, item2.price, item2.power
# those are similar to self.brand, self.price
class Refrigerator:
    # class object attribute (this attribute is global for class Refrigerator)
    seller = "Mr. Smith"
    
    def __init__(self, brand, price):
        self.brand = brand
        self.price = price
        self.guarantee = 3
        self.power = "electricity"
        
    def consumption(self):
        print("500 W")
        
    def description(self, company):
        print("Refrigerator {} is {} and the seller is {}. \nIt is under {}".format(self.brand, self.price, self.seller, company))
item = Refrigerator("LG", 200)
# running method within init
item.seller
# running method outside init --> use bracket ()
item.consumption(), item.description("LG Japan")
class Circle:
    
    # phi coefficient
    phi = 3.14
    
    def __init__(self, radius):
        self.radius = radius
        self.area = 2 * self.phi * (radius ** 2)
        
    def circumference(self):
        return 2 * self.phi * self.radius
    
#     def area(self):
#         return 2 * self.phi * (self.radius ** 2)
circle1 = Circle(10)
circle1.circumference(), circle1.area
# if the method used is outside init, then it should be "circle1.area()"
# parent class
class Student:
    
    status = "student"
    
    def __init__(self, name, classroom):
        self.name = name
        self.classroom = classroom
        
    def description(self):
        print("{} in classroom {} is a {}".format(self.name, self.classroom, self.status))
jack = Student("Jack", "12")
jack.description()
# child class --> this class inherits Student class
class Score(Student):
    
    def __init__(self, name, classroom):
        super().__init__(name, classroom)       # super will look for the similar def __init__ from parent class represents Student class
        # Student.__init__(name, classroom)     # it is also allowed, but it is recommended to use super()
        self.score_update = []
        
    def input_score(self, add):
        return self.score_update.append(add)
bill = Score("Bill", 13)
bill.description() # it is inherited from Student class
bill.input_score(90)
bill.score_update
class Cat:
    def __init__(self, name):
        self.name = name
        
    def response(self):
        return self.name + " miauw!"
    
class Dog:
    def __init__(self, name):
        self.name = name
        
    def response(self):
        return self.name + " guk-guk!"
bucky = Cat("Bucky")
kiku = Dog("Kiku")
bucky.response(), kiku.response()
for animals in (bucky, kiku):
    print(type(animals))
    print(animals.response())
def animal_speaks(animal):
    print(animal.response())
animal_speaks(kiku), animal_speaks(bucky)
class Sample:
    
    def __init__(self, name, number, word):
        self.name = name
        self.number = number
        self.word = word
        
    def print(self):
        return self.name
    
    def __str__(self):
        return self.name
sample = Sample('testing', 44, 'test')
jojo = Sample("Jojo", 50, "konami")
sample.print()
print(dir(sample))
sample.__str__(), str(sample) # can be used in two ways
jojo.print()
class BankAccount:
    
    def __init__(self, saving_amount):
        self.saving_amount = saving_amount

    def print_balance(self):
        print("Your balance is Rp {}".format(self.saving_amount))
    
    def save(self):
        add = int(input("input your amount to save = "))
        self.saving_amount += add
    
    def withdraw(self):
        substract = int(input("Enter the amount you want to witdraw = "))
        if self.saving_amount < substract:
            print("Sorry, your balance is not sufficient. \n Your balance is {}").format(self.saving_amount)
        else:
            self.saving_amount -= substract
mysaving = BankAccount(100000)
mysaving.print_balance()
mysaving.save()
mysaving.print_balance()
mysaving.withdraw()
mysaving.print_balance()
def christmas():
    print("We will have Christmas soon")
sample = christmas
sample, christmas
sample()
del christmas
christmas
sample, sample()
def easter(month="April"):
    print("Ready to welcome Easter")
    
    def location():
        return "\t the location is at our home"
  
    
    def confirmation():
        return "\n \t this is not April yet, check the calendar"
    
#     print(location())
#     print(confirmation())
    print("this is the last command in function easter")
    
    if month == "April":
        return location
    else:
        return confirmation
easter()
# function location doesnt appear since it is defined within function easter
check_easter = easter()
check_easter
check_easter()
print(check_easter())
check_more = easter("May")
print(check_more())
def greeting():
    return "Good morning"

greeting()
def news(other_function):
    print("I run another function")
    print(other_function)
news(greeting())
def mydecorator(function):
    def wrap_func():
        print("I am the command before the original function")
        print("\n")
        function()
        print("\n")
        print("I am the after original function")
        print("😄😄😄😄😄😄😄😄😄😄😄")
        
        
    return wrap_func
@mydecorator
def greeting():
    print("Good morning")
greeting()
def calculate():
    print("math and aritmathic should be mastered before learning AI and data science")
calculate()
@mydecorator
def calculate():
    print("math and aritmathic should be mastered before learning AI and data science")
calculate()
@mydecorator
def calculate():
    print("math and aritmathic should be mastered before learning AI and data science")
def greeting():
    print("Good morning")
    
# only the first function after @mydecorator will be affected
calculate()
greeting()
user1 = {'name': 'Barnes',
        'valid': True}
def authentication(function):
    def wrap_func(*args, **kwargs):
        if args[0]['valid']:
            return function(*args, **kwargs)
    return wrap_func
@authentication
def send_chat(user):
    print("chat is successfully sent")
send_chat(user1)
import sympy as sym
from IPython.display import Math
import math
import numpy as np
x
x = sym.symbols('x')
x
x + 1
x ** 7
display(x + 87)
y, z = sym.symbols('y, z')
x/y
display(Math("\\frac{x}{y}"))
display(sym.sqrt(6))
np.sqrt(6)
math.sqrt(6)
display(Math("\\sigma = \\frac {\\mu}{\\sqrt{x+y^{25z}}}"))
display(Math('z_{87k}'))
display(Math("\\text{the answer is} \\frac {x}{y^{15}}"))
display(Math('\\sin{3\\pi + \\theta}'))
display(Math('\\sin(3\\pi + \\theta)'))
display(Math('e = mc^2'))
mu, alpha, sigma = sym.symbols("mu, alpha, sigma")
p = mu + 2*45 + sym.exp(sigma+alpha)
p
x,y,z = sym.symbols("x,y,z")
p = x+10
p.subs(x, 2)
p = x + 10 + 4**y
p.subs({x:10,
       y:2})
x = sym.symbols('x')
p = 5*x + 5 - 20
sym.solve(p)
display(Math('\\text{the solution of case }%s \\text{ is x = } %g' %(sym.latex(p), sym.solve(p)[0])))
p2 = x**2 - 4
sym.solve(p2)
answer = sym.solve(p2)

for i in range(0, len(answer)):
    print("Solution #" + str(i+1) + " is " + str(answer[i]))
y = sym.symbols('y')
p3 = x/10 - x*y + 50
sym.solve(p3,y), sym.solve(p3,x)
import sympy as sym
from IPython.display import Math
x,y,z = sym.symbols('x y z')
p1 = x**y * x**z
display(p1)
display(sym.simplify(p1))
p2 = x**y / x**z
display(p2)
display(Math(sym.latex(p2)))
display(sym.simplify(p2))
p3 = x**y * y**z
display(p3)
display(sym.simplify(p3))
x,y,z = sym.symbols('x y z')
p1 = x*(x+1)
p2 = x
display(sym.expand(p1*p2))
p3 = y**12 + x**8
display(p1*p2*p3)
display(sym.expand(p1*p2*p3))
p4 = sym.sqrt(y)
display(p4)
display(sym.expand((p1*p2*p3)/p4))
display((p1*p2*p3)/p4)
display(sym.simplify(p1*p2*p3)/p4)
%whos
x,y,z = sym.symbols('x y z')
p1 = 5*(x*y)
p2 = (5*x)*y
p1-p2
display(p1)
display(p2)
a = sym.symbols('a')
x = a*(67-a) + 1/a**2 * (16+a)
display(x)
p6 = x*(y+z)
p7 = 3/x + x**2
display(p6)
display(p7)
display(p6*p7)
display(sym.expand(p6*p7))
display(p6*p7 - p6*p7)
b1 = [1,2,3,4,5]
sum(b1)
import numpy as np
np.sum(b1)
# products
np.prod(b1)
# cumulative summation
np.cumsum(b1)
b2 = np.array([[1, 2, 3],[4, 5, 6]])
print(b2)
display(Math(sym.latex(sym.simplify(b2))))
np.cumsum(b2)
b1
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.plot(b1, 'rs-')
plt.plot(np.cumsum(b1), 'bo-')
plt.legend(['b1', 'cumulative'])
plt.show();
plt.plot(b1, 'rs-')
plt.plot(np.cumprod(b1), 'bo-')
plt.legend(['b1', 'cumulative product'])
plt.show();
np.cumprod(b1)
# QUIZ 1
bil2 = np.arange(1,5)
p1 = np.sum(bil2)
p2 = np.sum(bil2**2)

j1 = p1/p2
j2 = 1/p1
print(f'Answer 1 is {j1} \nAnswer 2 is {j2}'), j1 == j2
# QUIZ 2
p3 = np.prod(bil2)
p4 = np.prod(bil2**2)

j3 = p3/p4
j4 = 1/p3
print(f'Answer 1 is {j3} \nAnswer 2 is {j4}'), j3 == j4