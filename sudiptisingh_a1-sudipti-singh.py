lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)
print(lst[0])
#b.	Print the last element of otherlst
print(otherlst[6])
#c.	Print the first five elements of lst
print(lst[0:5])
#d.	Print the fifth element of otherlst
print(otherlst[4])
#e.	Print the number of items in lst
print(len(lst))
#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[0:4])
#b.	Using indexing, print the substring "test" from s
print(s[10:14])
#c.	Print the contents of s starting from the 27th character (H)
print(s[27:])
#d.	Print the last three characters of s
print(s[31:])
#e.	Print the number of characters in s
print(len(s))
#importing "math" so that I can do any type of mathematical operation
import math
#calculates the factorial
math.factorial(13)
#prints the factorial
print(math.factorial(13))
#used 3 different variables that I named based on what kind of noun I thought best described them
feeling = "Happy"
condition = "New"
time = "Year"
#print the three variables
print(feeling, condition, time+"!")
#define three separate functions
def feeling(): 
    print("Happy")
def condition(): 
    print("New")
def time(): 
    print("Year"+"!")
#execute the functions
feeling()
condition()
time()
#define one function to include all the functions used before
def happy2020():
    #restate previous functions
    feeling()
    condition()
    time()
#execute the function
happy2020()
#define a function and call two parameters (x,y)
def basicAddition(x,y):
    print(x, '+', y, '=',x+y)
#execute the function
basicAddition(3,4)