lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 

otherlst = ['a','b','c','d','e','f','g']

s = "This is a test string for HCDE 530"



#Exercise 1 (working with a list):

print("Exercise 1")

#a.	Print the first element of lst (this one has been completed for you)

print(lst[0])

#b.	Print the last element of otherlst

print(otherlst[-1])

#c.	Print the first five elements of lst

print(otherlst[:5])

#d.	Print the fifth element of otherlst

print(otherlst[4])

#e.	Print the number of items in lst

print(len(lst))



#Exercise 2 (working with a string):

print("\nExercise 2")

#a.	Print the first four characters of s

print(s[:5])

#b.	Using indexing, print the substring "test" from s

print(s[10:15])

#c.	Print the contents of s starting from the 27th character (H)

print(s[26:])

#d.	Print the last three characters of s

print(s[len(s)-3:])

#e.	Print the number of characters in s

print(len(s))
def factorial(number):

    if(number >= 0): #factorials only work for numbers greater than or equal to 0

        y = 1 # initial value of any factorial

        x = 1 # use a counter

        while(x <= number): 

            y *= x #definition of a factorial is to multiple all numbers up to that number together

            x +=1 #count up until we reach the number we're trying to do a factorial for

    return y



factorial(13)
str1 = '\"Happy'

str2 = " New"

str3 =' Year!\"'



print(str1 + str2 + str3) # print the string all at once; could also use commas and remove the spaces in the strings, or even concatenation/append functions
def fun1():

    str1 = "Happy"

    return str1

def fun2():

    str2 = "New"

    return str2

def fun3():

    str3 = 'Year!'

    return str3



print(fun1()) #print displays output

print(fun2())

print(fun3())
def fun1():

    str1 = "Happy"

    return str1

def fun2():

    str2 = "New"

    return str2

def fun3():

    str3 = 'Year!'

    return str3

def combofun():

    print(fun1(),fun2(),fun3()) # do the same thing as previous, but combine function calls so we have a single printline



combofun()
def addition(num1, num2): #set token names for function

    sum = num1 + num2 # get the output value

    str1 = str(num1) + ' + ' + str(num2) + ' = ' + str(sum) #display the string form of the addition problem

    return str1



addition(3,4)