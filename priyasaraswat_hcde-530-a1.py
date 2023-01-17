lst = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
otherlst = ['a','b','c','d','e','f','g']
s = "This is a test string for HCDE 530"

#Exercise 1 (working with a list):
#a.	Print the first element of lst (this one has been completed for you)
print(lst[0])
#b.	Print the last element of otherlst
print(otherlst[6])
#c.	Print the first five elements of lst
print(lst[0:6])
#d.	Print the fifth element of otherlst
print(otherlst[4])
#e.	Print the number of items in lst
print(len(lst))

#Exercise 2 (working with a string):
#a.	Print the first four characters of s
print(s[0:4])
#b.	Using indexing, print the substring "test" from s
newlist = s.split()
print(newlist[3])
#c.	Print the contents of s starting from the 27th character (H)
print(s[26:])
#d.	Print the last three characters of s
print(s[31:])
#e.	Print the number of characters in s
print(len(s))
#Defined the variable to calculate the factorial
x = 13
#Defined the variable to control the looping
y = 13
while (y>1):
    #The code will multiply the each outcome with the previous number in the series
    x = x*(y-1)
    y = y - 1
print("13! = ",x)
#The strings are defined as independent variables
s1 = 'Happy'
s2 = 'New'
s3 = 'Year!'
#Combined the values to generate the desired outcome
print(s1,s2,s3)
#Three independent functions are defined to return one string in Happy New Year!
def s1():
    x = 'Happy'
    return x
def s2():
    y = 'New'
    return y
def s3():
    z = 'Year!'
    return z

#All functions are called to print the outcome
print(s1())
print(s2())
print(s3())
# New function to print the outcome
def snew(t,r,q):
    print (t,r,q)

# Values from the previous function is passed to the new function
snew(s1(),s2(),s3())    

# Function to add two numbers and print the output 
def addnum(x,y):
    z = x + y
    print(x,'+',y,'=',z)

# Passing values to the function
addnum(10,11)