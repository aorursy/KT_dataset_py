#### lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
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
print(s[0:4])
#b.	Using indexing, print the substring "test" from s
print(s[10:14])
#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])
#d.	Print the last three characters of s
print(s[-3:])
#e.	Print the number of characters in s
print(len(s))
# Python program to find the factorial of 13

#defined a variable to store the value
fact=13

#defined a variable to store the multiplication value
y=13
while(fact > 1):
  y = y * (fact - 1)
  fact = fact - 1
    
#printing the result
print ('Factorial of 13 is', y)
# Python program to print "Happy New Year!" using three different string variables

# defined variable with string values
s1 = 'Happy'
s2 = 'New'
s3 = 'Year!'

#printing the result
print(s1, s2, s3)
# Python program to define three functions that each return one string of "Happy", "New", and "Year!"

# Defined a function to return "Happy"
def s1():
    a = 'Happy'
    return a

# Defined a function to return "New"
def s2():
    b = 'New'
    return b

# Defined a function to return "Year!"
def s3():
    c = 'Year!'
    return c

#printing the result by executing the three functions
print(s1())
print(s2())
print(s3())
# Python program to write a function using the above created functions to print "Happy New Year!"

# Defined a function to take three parameters
def s0(a,b,c):
    print (a,b,c)

# Passesd the values of the functions defined in the prior problem as parameters
s0(s1(),s2(),s3())
# Python program to write a function that takes two parameters and adds them together

# Defined a function to take two parameters
def add(x, y):
    
    # Defined a variable to store the sum of two numbers
    sum = x + y
    
    # Printing the result
    print(x, "+", y, "=", sum)
    
# Executing the defined function with two parameters    
add(2, 3)   