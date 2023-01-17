#this is a comment

#printing hello world

print("Hello World")
#Using forward, pipe and underscore slash to draw a triangle

print("    /|")

print("   / |")

print("  /  |")

print(" /   |")

print("/____|")
#Using variables in python

#printing a story using python

print("I saw a man named john")

print("he was 67 and white")

print("john loved to play with his grand son kattie")

print("but he didnt like being 67")
#Like this

print("I saw a man named Mike")

print("he was 67 and white")

print("mike loved to play with his grand son kattie")

print("but he didnt like being 67")



#but what if i had a long long story say about 20 pages or that 480
#creating variable

#variable_name = "vale_of_variable"

person_Name = "Mike"

person_Age = "57"

print("---------------------------------------------")

#usage : wrapping what to be printed in "" add a pluse sign + and call the variable + "statement"

print("I saw a man named " + person_Name + ",")
print("I saw a man named " + person_Name + ",")

print("he was " + person_Age + " and white")

print( person_Name + " loved to play with his grand son kattie")

print("but he didnt like being " + person_Age + ".")
print("I saw a man named " + person_Name + ",")

print("he was " + person_Age + " and white")

print("---------------------------------------------")

#changing name to beth and age to 60

person_Name = "Beth"

person_Age = "60"

print(""+ person_Name + " loved to play with his grand son kattie")

print("but he didnt like being " + person_Age + ".")
#storing string as variables

#use Qoutation mark: "string Text goes here"

print("learn python")

print("---------------------------------------------")

#create new line in the string "Learn\npython"

print("learn\npython")
phrase = "Learn Python"

print(phrase)

print(phrase + "is cool")
print(phrase.lower())

print(phrase.upper())

print(phrase.upper().isupper())

print(phrase.isupper())

print(len(phrase))

print(phrase.replace("Learn","Awesome"))
print(phrase[0])

print(phrase[1])

print(phrase[6])

print(phrase[10])

print(phrase[5])

print(phrase[3])

print("---------------------------------------------")

#printing the location

print(phrase.index("L"))

print(phrase.index("P"))

print(phrase.index("e"))

print(phrase.index("t"))

print(phrase.index("y"))
#ptinting a number

print(2)

#or decimals

print(2.333)

#also negative numbers

print(-234.8)
print(3+2)

print(3-2)

print(3*2)

#complex eqations using paranthesis

print(3*2 + (56-89)-(6+2))

# A mod function

print(22%2)

#power functio

print(pow(4,5))

#min/max

print(max(4,5))

print(min(4,5))

#round

print(round(-5.6))

#this prints -6
my_num = 5

print(5 + "printing digit")
my_num = 5

print("5" + " printing digit")
from math import *

#math module gives us access to lot of different math function

my_num = -5

print(sqrt(36))
name = input("Enter your name:")

age = input("Enter your age:")

print("Hello " + name + "! you are " + age)
num1 = input("Enter a number: ")

num2 = input("Enter another number: ")

result = num1 + num2

print(result)

print("---------------------------------------------")

# gets 34 which is wrong this is because python things its concatenation of two strings

#using int() function to covertthese into numbers if using decimal use float() rather than int()

num3 = input("Enter a number: ")

num4 = input("Enter another number: ")

result = int(num3) + int(num4)

print(result)

color = input("Enter a color:")

plural_noun = input("Enter a plural_noun:")

celebrity = input("Enter a celebrity")

print("Roses are" + color)

print(plural_noun +"are blue")

print("I love "+ celebrity)
#create a list

# lets give a discriptive name

deserts = ["Namib Desert","Atacama Desert","Sahara Desert","Gobi Desert","Mojave Desert" ]

print("---------------------------------------------")

#print the whole list

print(deserts)

print("---------------------------------------------")

#print specific elements 

#use index to print the specific elements in the list use [] with the list name

print(deserts[4])

#index from the back of the list

print(deserts[-1])

#select the portion of the list

print("---------------------------------------------")

print(deserts[1:3])

print(deserts[1:])

print(deserts[:3])

#Access 
lucky_numbers = [4,5,6,6,7,8,2,2,3,3]

friends = ["mike","john", "lara", "anne"]

# 1 Printing all elements of the list

print(lucky_numbers)

print(friends)

print("---------------------------------------------")

#extend() to addon to the list

friends.extend(lucky_numbers)

print(friends)

print("---------------------------------------------")

#add individual elements to the list

friends = ["mike","john", "lara", "anne"]

friends.append("Karen")

print(friends)

print("---------------------------------------------")

#insert() 

friends.insert(1,"kelly")

print(friends)

print("---------------------------------------------")

friends.insert(4,"nike")

print(friends)

print("---------------------------------------------")

friends.clear()

print(friends)

coordinate = (6,7)

print(coordinate[1])
# creating a function to say hi to a variablie

def say_hi():

    print("Hello user")

say_hi()

print("---------------------------------------------")

#parameters that can be passed in to the function

def sayhi(name):

    print(" Hello " + name)    

sayhi("Mike")

print("---------------------------------------------")

print("MAKE FUCTIONS POWERFULL BY GIVING INFORMATION")

def sayhi(name, age):

    print("Hello " + name + " you are " + age)    

sayhi("Mike", "age")
# return keyword

def cube1(x):

    y = x*x*x

    return y

print(cube(4))



def cube2(x):

    return x*x*x

print(cube2(9))
def cube2(x):

    return x*x*x

result = cube2(9)

print(result)
# simple statement

is_male = True

if is_male:

    print("you are a male")

    

print("---------------------------------------------")

is_female = False

if is_female:

    print("you are a female")

else:

    print("you are a male")
def max_num(num1,num2,num3):

    if num1 >= num2 and num1 >= num3:

        return num2

    elif num2 >= num1 and num2 >= num3:

        return num2

    else:

        return num3

    

print(max_num(3,4,5))
#input from the user and convert the numbers into string with float()

num1 = float(input("Enter first number:"))

op = input("Enter Operator:")

num2 = float(input("Enter second number:"))



if op == "+":

    print(num1 + num2)

elif op == "-":

    print(num1 - num2)

if op == "*":

    print(num1 * num2)

elif op == "/":

    print(num1 / num2)

elif op == "%":

    print(num1 % num2)

else:

    print("input invald")



try:

    number = int(input("Enter a number: "))

    print(number)

except:

    print("Invalid Input")
i = 1

while i <= 10:

    print(i)

    i += 1

print("Done with loop")