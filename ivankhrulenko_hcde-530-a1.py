lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 

otherlst = ['a','b','c','d','e','f','g']

s = "This is a test string for HCDE 530"



#Exercise 1 (working with a list):

#a.	Print the first element of lst (this one has been completed for you)

print(lst[0])

#b.	Print the last element of otherlst

print(otherlst[-1])

#c.	Print the first five elements of lst

print(lst[:5])

#d.	Print the fifth element of otherlst

print(otherlst[4])

#e.	Print the number of items in lst

print(len(lst))

#Exercise 2 (working with a string):

#a.	Print the first four characters of s

print(s[:4])

#b.	Using indexing, print the substring "test" from s

print(s[10:15])

#c.	Print the contents of s starting from the 27th character (H)

print(s[26:])

#d.	Print the last three characters of s

print(s[-3:])

#e.	Print the number of characters in s

print(len(s))
def factorial_fun(x):

    result=x

    i=x-1

    while i>0:

        result*=i

        i-=1

    return result

factorial_fun(13)
happy = "Happy"

new = "New"

year = "Year!"

print (happy, new, year)
def happy():

    return "Happy"

def new():

    return "New"

def year():

    return "Year!"

#not sure if it is what's expected

print(happy())

print(new())

print(year())
def printHNY():

    print(happy(),new(),year())

printHNY()
def addition(a,b):

    print (a,"+",b,"=",a+b)

addition(3,4)